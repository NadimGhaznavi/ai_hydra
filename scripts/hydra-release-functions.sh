#!/usr/bin/env bash
# hydra-release-functions.sh

# (optional) guard against double-sourcing
[[ ${_HELPER_SH_LOADED:-} ]] && return
_HELPER_SH_LOADED=1

DIV="----------------------------------------------------------------------------"

dev_branch_process() {
	# Print out the current git status
	echo "Current status of the git repository: git status..."
	git status
	echo $DIV

	# Print out current branches
	echo "Current branches..."
	git branch
	echo $DIV

	while true; do
		read -rp "Create a new feature branch? [y|n]: " ANSWER
		ANSWER=${ANSWER,,} # lowercase

		[[ "$ANSWER" == y || "$ANSWER" == n ]] || {
			echo "Enter y or n"
			continue
		}

		[[ "$ANSWER" == y ]] && {
			ACTION="new_branch"
			break
		}
		[[ "$ANSWER" == n ]] && {
			ACTION="exit"
			break
		}
	done

	if [[ "$ACTION" == "exit" ]]; then
		echo "Abort count down, exiting now..."
		return 1
	fi

	while true; do
		read -rp "Enter new branch name (e.g. feat/widget): " NEW_BRANCH
		[[ -n "$NEW_BRANCH" ]] || {
			echo "Branch cannot be empty"
			continue
		}
		break
	done

	# Create new breanch
	check_clean_git
	git checkout -b "$NEW_BRANCH"
}

feat_branch_process() {
	# Print out git status
	echo "Confirm that git state is correct:"
	echo $DIV
	git status
	echo $DIV

	while true; do
		read -rp "Ready to release from $CUR_BRANCH [y|n]: " ANSWER
		ANSWER=${ANSWER,,} # lowercase

		[[ "$ANSWER" == y || "$ANSWER" == n ]] || {
			echo "Enter y or n"
			continue
		}

		[[ "$ANSWER" == y ]] && {
			ACTION="release"
			break
		}
		[[ "$ANSWER" == n ]] && {
			ACTION="exit"
			break
		}
	done

	if [[ "$ACTION" == "exit" ]]; then
		echo "Abort count down, exiting now..."
		return 1
	fi

	# Get the version out of the TOML file
	TOML_VERSION="$(get_cur_toml_version $BASE_DIR)"
	echo "Current TOML file version    : $TOML_VERSION"

	# Get the version out of the Constants file
	CONST_VERSION="$(get_cur_const_version $BASE_DIR)"
	echo "Current Constants version    : $CONST_VERSION"

	# Get the version out of the docs/_source/conf.py file
	DOCS_VERSION="$(get_cur_docs_version $BASE_DIR)"
	echo "Current docs/_source/conf,py : $DOCS_VERSION"

	# Get the version out of the module __init__.py file
	#INIT_VERSION="$(get_cur_init_version $BASE_DIR)"
	#echo "Current __init__.py version  : $INIT_VERSION"

	echo $DIV

	NEW_VERSION="$(get_new_version)"
	NEW_RELEASE_STR="$(get_new_release_name)"

	echo "New version set to          : $NEW_VERSION"
	echo "Release string              : $NEW_RELEASE_STR"

	# Get this feature branch name
	FEAT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"

	echo $DIV

	# Switch the the dev branch
	echo "Switching to the dev branch..."
	check_clean_git
	git checkout dev
	echo $DIV

	# Merge the feature branch into dev
	echo "Merging the feature branch ($FEAT_BRANCH) into dev..."
	git merge "$FEAT_BRANCH"
	echo $DIV

	# Create a new branch for the release
	echo "Creating a new release branch: release/$NEW_VERSION..."
	check_clean_git
	git checkout -b "release/$NEW_VERSION"
	echo $DIV

	# Update the version number in the TOML file
	echo "Updating the version in the pyproject.toml file..."
	update_toml_version "$NEW_VERSION"
	echo $DIV

	# Update the version number in the docs file
	echo "Updating the release number in the docs/_source/conf.py..."
	update_docs_version "$NEW_VERSION"
	echo $DIV

	# Update the version number in the constants file
	echo "Updating the version in the DHydra constants file..."
	update_constants_version "$NEW_VERSION"
	echo $DIV

	# Update the version number in the __init__.py file
	#echo "Updating the version in the project __init__.py file..."
	#update_init_version "$NEW_VERSION"
	#echo $DIV

	# Add and commit the updated files
	echo "Add and commit with git..."
	git add -v pyproject.toml ai_hydra/constants/DHydra.py docs/_source/conf.py ai_hydra/__init__.py
	git commit -m "Pre $NEW_RELEASE_STR"
	git push -u origin "release/$NEW_VERSION"
	echo $DIV

	# Switch the the main branch
	echo "Switching to the main branch..."
	check_clean_git
	git checkout main
	echo $DIV

	# Merge the release into main
	echo "Merge release/$NEW_VERSION into main..."
	git merge "release/$NEW_VERSION"
	echo $DIV

	# Tag the release
	echo "Tagging the repo contents..."
	git tag -a "v$NEW_VERSION" -m "$NEW_RELEASE_STR"
	echo $DIV

	# Pushing the new, tagged release to GitHub
	echo "Pushing new release ($NEW_RELEASE_STR) to GitHub..."
	git push origin main --follow-tags
	echo $DIV

	# Switch back to the DEV branch
	echo "Switching back to the dev branch"
	check_clean_git
	git checkout dev
	echo $DIV

	# Merge the new release back into dev
	echo "Merging the new release/$NEW_VERSION back into dev..."
	git merge "release/$NEW_VERSION"
	echo $DIV

	# Push the changes into the remote dev repo
	echo "Synching updates with GitHub"
	git add . -v
	git commit -m "Merged $NEW_VERSION back into dev"
	echo $DIV

	echo "🚀 Successful release!!!"

	# We're on the dev branch, run the dev_branch_process to get back to a feature
	# branch
	dev_branch_process
}

get_base_dir() {
	local scripts_dir
	scripts_dir="$(get_scripts_dir)"
	(cd -- "$scripts_dir/.." && pwd)
}

check_clean_git() {
	git diff --quiet || {
		echo "ERROR: working tree dirty"
		git status
		exit 1
	}
	git diff --cached --quiet || {
		echo "ERROR: index has staged changes"
		git status
		exit 1
	}
}

get_cur_const_version() {
	local BASE_DIR="$1"
	CONST_FILE=$BASE_DIR/ai_hydra/constants/DHydra.py
	CONST_VERSION="$(sed -nE 's/^[[:space:]]*VERSION[[:space:]]*:[[:space:]]*str[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' "$CONST_FILE" | head -n1)"
	[[ -n "${CONST_VERSION:-}" ]] || {
		echo "ERROR: Could not find VERSION in $CONST_FILE" >&2
		exit 1
	}
	echo $CONST_VERSION
}

get_cur_docs_version() {
	local BASE_DIR="$1"
	DOCS_FILE="$BASE_DIR/docs/_source/conf.py"

	DOCS_VERSION="$(
		sed -nE 's/^[[:space:]]*release[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' "$DOCS_FILE" |
			head -n1
	)"

	[[ -n "${DOCS_VERSION:-}" ]] || {
		echo "ERROR: Could not find release version in $DOCS_FILE" >&2
		exit 1
	}

	echo "$DOCS_VERSION"
}

get_cur_toml_version() {
	local BASE_DIR="$1"
	# Get the version out of the TOML file
	TOML_FILE=$BASE_DIR/pyproject.toml
	TOML_VERSION="$(sed -nE 's/^[[:space:]]*version[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' "$TOML_FILE" | head -n1)"
	[[ -n "${TOML_VERSION:-}" ]] || {
		echo "ERROR: Could not find version in $TOML_FILE" >&2
		exit 1
	}
	echo $TOML_VERSION
}

get_cur_init_version() {
	local BASE_DIR="$1"
	INIT_FILE="$BASE_DIR/ai_hydra/__init__.py"
	INIT_VERSION="$(sed -nE 's/^[[:space:]]*__version__[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' "$INIT_FILE" | head -n1)"
	[[ -n "${INIT_VERSION:-}" ]] || {
		echo "ERROR: Could not find __version__ in $INIT_FILE" >&2
		exit 1
	}
	echo $INIT_VERSION
}

get_new_release_name() {
	while true; do
		read -rp "Enter new release name: " NEW_RELEASE_NAME

		[[ -n "$NEW_RELEASE_NAME" ]] || {
			echo "Releases must be named."
			continue
		}

		break
	done
	NEW_RELEASE_STR="Release v$NEW_VERSION - $NEW_RELEASE_NAME"
	echo $NEW_RELEASE_STR
}

get_new_version() {
	while true; do
		read -rp "Enter new version number (e.g. 0.4.6): " NEW_VERSION

		[[ -n "$NEW_VERSION" ]] || {
			echo "Version cannot be empty."
			continue
		}

		[[ "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || {
			echo "Invalid version format (expected X.Y.Z)."
			continue
		}

		break
	done
	echo $NEW_VERSION
}

get_scripts_dir() {
	echo "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
}

main_branch_proc() {
	local BASE_DIR="$1"
	echo
	echo "Base directory            : $BASE_DIR"

	# Get the version out of the TOML file
	TOML_VERSION=$(get_cur_toml_version $BASE_DIR)
	echo "Current TOML file version : $TOML_VERSION"

	# Get the version out of the Constants file
	CONST_VERSION=$(get_cur_const_version $BASE_DIR)
	echo "Current Constants version : $CONST_VERSION"

	echo

	NEW_VERSION=$(get_new_version)
	NEW_RELEASE_STR=$(get_new_release_name)

	echo "New version set to : $NEW_VERSION"
	echo "Release string     : $NEW_RELEASE_STR"
}

update_constants_version() {
	local NEW_VERION="$1"
	BASE_DIR=$(get_base_dir)
	CONSTANTS_FILE="$BASE_DIR/ai_hydra/constants/DHydra.py"
	if [ -f "$CONSTANTS_FILE" ]; then
		# Use sed to replace the VERSION constant
		if [[ "$OSTYPE" == "darwin"* ]]; then
			# macOS sed requires -i ''
			sed -i '' "s/VERSION: str = \".*\"/VERSION: str = \"$NEW_VERION\"/" "$CONSTANTS_FILE"
		else
			# Linux sed
			sed -i "s/VERSION: str = \".*\"/VERSION: str = \"$NEW_VERION\"/" "$CONSTANTS_FILE"
		fi
		echo "✅ Updated $CONSTANTS_FILE VERSION to $NEW_VERION"
	else
		echo "❌ Error: $CONSTANTS_FILE not found"
		exit 1
	fi
}

update_docs_version() {
	local NEW_VERSION="$1"
	BASE_DIR=$(get_base_dir)
	DOCS_FILE="$BASE_DIR/docs/_source/conf.py"
	if [ -f "$DOCS_FILE" ]; then
		# Use sed to replace the VERSION constant
		if [[ "$OSTYPE" == "darwin"* ]]; then
			# macOS sed requires -i ''
			sed -i '' "s/release = \".*/release = \"$NEW_VERION\"/" "$DOCS_FILE"
		else
			# Linux sed
			sed -i "s/^release = \".*\"/release = \"$NEW_VERSION\"/" $DOCS_FILE
		fi
		echo "✅ Updated $DOCS_FILE VERSION to $NEW_VERION"
	else
		echo "❌ Error: $DOCS_FILE not found"
		exit 1
	fi

}

update_init_version() {
	local NEW_VERSION="$1"
	BASE_DIR=$(get_base_dir)
	INIT_FILE="$BASE_DIR/ai_hydra/__init__.py"
	if [ -f "$INIT_FILE" ]; then
		# Use sed to replace the __version__ variable
		if [[ "$OSTYPE" == "darwin"* ]]; then
			# macOS sed requires -i ''
			sed -i '' "s/^__version__ = \".*\"/__version__ = \"$NEW_VERSION\"/" "$INIT_FILE"
		else
			# Linux sed
			sed -i "s/^__version__ = \".*\"/__version__ = \"$NEW_VERSION\"/" "$INIT_FILE"
		fi
		echo "✅ Updated $INIT_FILE __version__ to $NEW_VERSION"
	else
		echo "❌ Error: $INIT_FILE not found"
		exit 1
	fi
}

update_toml_version() {
	local NEW_VERSION="$1"
	BASE_DIR=$(get_base_dir)
	TOML_FILE="$BASE_DIR/pyproject.toml"

	if [ -f "$TOML_FILE" ]; then
		# Use sed to replace the version line
		if [[ "$OSTYPE" == "darwin"* ]]; then
			# macOS sed requires -i ''
			sed -i '' "s/^version = \".*\"/version = \"$NEW_VERSION\"/" $TOML_FILE
		else
			# Linux sed
			sed -i "s/^version = \".*\"/version = \"$NEW_VERSION\"/" $TOML_FILE
		fi
		echo "✅ Updated $TOML_FILE version to $NEW_VERSION"
	else
		echo "❌ Error: $TOML_FILE not found"
		exit 1
	fi
}
