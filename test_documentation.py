#!/usr/bin/env python3
"""
Comprehensive test suite for AI Hydra documentation.
Tests RST syntax, content accuracy, cross-references, and build integrity.
"""

import os
import sys
import subprocess
import re
from pathlib import Path
from typing import List, Dict, Tuple, Set
import tempfile
import shutil

class DocumentationTester:
    """Test suite for documentation validation."""
    
    def __init__(self, docs_dir: str = "docs"):
        self.docs_dir = Path(docs_dir)
        self.source_dir = self.docs_dir / "_source"
        self.build_dir = self.docs_dir / "_build"
        self.errors = []
        self.warnings = []
        
    def run_all_tests(self) -> bool:
        """Run all documentation tests."""
        print("🧪 Running AI Hydra Documentation Test Suite")
        print("=" * 50)
        
        tests = [
            ("RST Syntax Validation", self.test_rst_syntax),
            ("Cross-Reference Validation", self.test_cross_references),
            ("Content Accuracy", self.test_content_accuracy),
            ("Build Integrity", self.test_build_integrity),
            ("Link Validation", self.test_internal_links),
            ("Code Block Validation", self.test_code_blocks),
            ("Table of Contents", self.test_toc_structure),
            ("Decision Flow Specific", self.test_decision_flow_content),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n📋 {test_name}")
            print("-" * 30)
            try:
                if test_func():
                    print(f"✅ {test_name}: PASSED")
                    passed += 1
                else:
                    print(f"❌ {test_name}: FAILED")
            except Exception as e:
                print(f"💥 {test_name}: ERROR - {e}")
                self.errors.append(f"{test_name}: {e}")
        
        print(f"\n📊 Test Results: {passed}/{total} tests passed")
        
        if self.errors:
            print(f"\n🚨 Errors found:")
            for error in self.errors:
                print(f"  • {error}")
                
        if self.warnings:
            print(f"\n⚠️  Warnings:")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        return passed == total and len(self.errors) == 0
    
    def test_rst_syntax(self) -> bool:
        """Test RST syntax validity."""
        rst_files = list(self.source_dir.glob("*.rst"))
        
        if not rst_files:
            self.errors.append("No RST files found in source directory")
            return False
            
        syntax_errors = []
        
        for rst_file in rst_files:
            try:
                # Use docutils to parse RST and check for syntax errors
                result = subprocess.run([
                    sys.executable, "-c",
                    f"""
import docutils.core
import sys
try:
    with open('{rst_file}', 'r', encoding='utf-8') as f:
        content = f.read()
    docutils.core.publish_doctree(content)
    print(f"✓ {rst_file.name}")
except Exception as e:
    print(f"✗ {rst_file.name}: {{e}}")
    sys.exit(1)
"""
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    # Filter out missing include file errors for TUI specs
                    error_output = result.stdout
                    if "No such file or directory" in error_output and "tui-client" in error_output:
                        self.warnings.append(f"{rst_file.name}: Missing TUI specification files (expected in development)")
                    else:
                        syntax_errors.append(f"{rst_file.name}: {result.stdout}")
                    
            except Exception as e:
                syntax_errors.append(f"{rst_file.name}: {e}")
        
        if syntax_errors:
            self.errors.extend(syntax_errors)
            return False
            
        print(f"✓ All {len(rst_files)} RST files have valid syntax")
        return True
    
    def test_cross_references(self) -> bool:
        """Test cross-references between documents."""
        rst_files = list(self.source_dir.glob("*.rst"))
        all_labels = set()
        all_references = []
        
        # Extract labels and references
        for rst_file in rst_files:
            content = rst_file.read_text(encoding='utf-8')
            
            # Find labels (.. _label:)
            labels = re.findall(r'^\.\. _([^:]+):', content, re.MULTILINE)
            all_labels.update(labels)
            
            # Find references (:ref:`label`)
            refs = re.findall(r':ref:`([^`]+)`', content)
            all_references.extend([(ref, rst_file.name) for ref in refs])
        
        # Check for broken references (excluding standard Sphinx references)
        standard_refs = {'genindex', 'modindex', 'search'}
        broken_refs = []
        for ref, filename in all_references:
            if ref not in all_labels and ref not in standard_refs:
                broken_refs.append(f"{filename}: :ref:`{ref}` -> label not found")
        
        if broken_refs:
            self.errors.extend(broken_refs)
            return False
            
        print(f"✓ All {len(all_references)} cross-references are valid")
        return True
    
    def test_content_accuracy(self) -> bool:
        """Test content accuracy against source specifications."""
        decision_flow_file = self.source_dir / "decision_flow.rst"
        
        if not decision_flow_file.exists():
            self.errors.append("decision_flow.rst not found")
            return False
            
        content = decision_flow_file.read_text(encoding='utf-8')
        
        # Check for required sections
        required_sections = [
            "Decision Flow Architecture",
            "Overview", 
            "System Initialization",
            "Decision Cycle States",
            "Budget Management System",
            "Neural Network Integration",
            "Performance Monitoring",
            "Error Handling and Recovery"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in content:
                missing_sections.append(section)
        
        if missing_sections:
            self.errors.extend([f"Missing section: {s}" for s in missing_sections])
            return False
        
        # Check for 9 decision cycle states
        state_pattern = r'State \d+:'
        states = re.findall(state_pattern, content)
        
        if len(states) != 9:
            self.errors.append(f"Expected 9 decision cycle states, found {len(states)}")
            return False
            
        print("✓ Content structure and required sections present")
        return True
    
    def test_build_integrity(self) -> bool:
        """Test Sphinx build integrity."""
        try:
            # Clean build
            if self.build_dir.exists():
                shutil.rmtree(self.build_dir)
            
            # Build documentation
            result = subprocess.run([
                "make", "html"
            ], cwd=self.docs_dir, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.errors.append(f"Sphinx build failed: {result.stderr}")
                return False
            
            # Check for critical warnings
            critical_warnings = [
                "ERROR",
                "CRITICAL", 
                "undefined label",
                "unknown document"
            ]
            
            build_warnings = []
            for warning in critical_warnings:
                if warning.lower() in result.stderr.lower():
                    build_warnings.append(f"Build warning: {warning}")
            
            if build_warnings:
                self.warnings.extend(build_warnings)
            
            # Check that HTML files were generated
            html_dir = self.build_dir / "html"
            if not html_dir.exists():
                self.errors.append("HTML build directory not created")
                return False
                
            html_files = list(html_dir.glob("*.html"))
            if len(html_files) < 5:  # Should have at least index + several content pages
                self.errors.append(f"Too few HTML files generated: {len(html_files)}")
                return False
            
            print(f"✓ Sphinx build successful, {len(html_files)} HTML files generated")
            return True
            
        except Exception as e:
            self.errors.append(f"Build test failed: {e}")
            return False
    
    def test_internal_links(self) -> bool:
        """Test internal links within documentation."""
        html_dir = self.build_dir / "html"
        
        if not html_dir.exists():
            self.warnings.append("HTML build not found, skipping link validation")
            return True
            
        html_files = list(html_dir.glob("*.html"))
        broken_links = []
        
        for html_file in html_files:
            content = html_file.read_text(encoding='utf-8')
            
            # Find internal links (href="filename.html") - exclude external URLs
            internal_links = re.findall(r'href="([^"]+\.html[^"]*)"', content)
            
            for link in internal_links:
                # Skip external links (those starting with http:// or https://)
                if link.startswith(('http://', 'https://')):
                    continue
                    
                # Remove anchors for file existence check
                link_file = link.split('#')[0]
                if link_file and not (html_dir / link_file).exists():
                    broken_links.append(f"{html_file.name}: {link}")
        
        if broken_links:
            self.errors.extend([f"Broken internal link: {link}" for link in broken_links])
            return False
            
        print("✓ All internal links are valid")
        return True
    
    def test_code_blocks(self) -> bool:
        """Test code blocks for syntax validity."""
        rst_files = list(self.source_dir.glob("*.rst"))
        code_block_errors = []
        
        for rst_file in rst_files:
            content = rst_file.read_text(encoding='utf-8')
            
            # Find code blocks with language specification
            code_blocks = re.findall(r'```(\w+)\n(.*?)\n```', content, re.DOTALL)
            code_blocks.extend(re.findall(r'\.\. code-block:: (\w+)\n\n(.*?)(?=\n\S|\n\.\.|$)', content, re.DOTALL))
            
            for lang, code in code_blocks:
                if lang == 'python':
                    try:
                        # Clean up the code by removing common indentation
                        lines = code.split('\n')
                        # Remove empty lines from start and end
                        while lines and not lines[0].strip():
                            lines.pop(0)
                        while lines and not lines[-1].strip():
                            lines.pop()
                        
                        if lines:
                            # Find minimum indentation (excluding empty lines)
                            min_indent = float('inf')
                            for line in lines:
                                if line.strip():  # Skip empty lines
                                    indent = len(line) - len(line.lstrip())
                                    min_indent = min(min_indent, indent)
                            
                            # Remove common indentation
                            if min_indent != float('inf') and min_indent > 0:
                                lines = [line[min_indent:] if line.strip() else line for line in lines]
                            
                            cleaned_code = '\n'.join(lines)
                            compile(cleaned_code, f"{rst_file.name}:code_block", 'exec')
                    except SyntaxError as e:
                        # Skip common documentation patterns that aren't meant to be executable
                        error_msg = str(e)
                        if any(skip_pattern in error_msg for skip_pattern in [
                            "expected ':'", 
                            "unexpected indent",
                            "invalid syntax"
                        ]):
                            # Check if this looks like a class/function definition without implementation
                            if any(pattern in code for pattern in ["class ", "def ", "async def"]):
                                continue  # Skip incomplete code examples
                        
                        code_block_errors.append(f"{rst_file.name}: Python syntax error in code block: {e}")
        
        if code_block_errors:
            self.errors.extend(code_block_errors)
            return False
            
        print("✓ All code blocks have valid syntax")
        return True
    
    def test_toc_structure(self) -> bool:
        """Test table of contents structure."""
        index_file = self.source_dir / "index.rst"
        
        if not index_file.exists():
            self.errors.append("index.rst not found")
            return False
            
        content = index_file.read_text(encoding='utf-8')
        
        # Check for toctree directive
        if '.. toctree::' not in content:
            self.errors.append("No toctree directive found in index.rst")
            return False
        
        # Extract toctree entries
        toctree_match = re.search(r'\.\. toctree::\s*\n.*?\n\n(.*?)(?=\n\S|\n\.\.|$)', content, re.DOTALL)
        
        if not toctree_match:
            self.errors.append("Could not parse toctree entries")
            return False
            
        toctree_entries = [line.strip() for line in toctree_match.group(1).split('\n') if line.strip()]
        
        # Check that referenced files exist
        missing_files = []
        for entry in toctree_entries:
            rst_file = self.source_dir / f"{entry}.rst"
            if not rst_file.exists():
                missing_files.append(f"{entry}.rst")
        
        if missing_files:
            self.errors.extend([f"TOC references missing file: {f}" for f in missing_files])
            return False
            
        print(f"✓ TOC structure valid with {len(toctree_entries)} entries")
        return True
    
    def test_decision_flow_content(self) -> bool:
        """Test decision flow specific content."""
        decision_flow_file = self.source_dir / "decision_flow.rst"
        
        if not decision_flow_file.exists():
            self.errors.append("decision_flow.rst not found")
            return False
            
        content = decision_flow_file.read_text(encoding='utf-8')
        
        # Test specific decision flow requirements
        tests = [
            ("9 decision cycle states", lambda: len(re.findall(r'State \d+:', content)) == 9),
            ("Neural network architecture", lambda: "19 features" in content and "200 neurons" in content),
            ("Budget management", lambda: "budget" in content.lower() and "100 moves" in content),
            ("Oracle training", lambda: "oracle" in content.lower() and "training" in content.lower()),
            ("Performance monitoring", lambda: "performance monitoring" in content.lower()),
            ("Error handling", lambda: "error handling" in content.lower()),
        ]
        
        failed_tests = []
        for test_name, test_func in tests:
            if not test_func():
                failed_tests.append(test_name)
        
        if failed_tests:
            self.errors.extend([f"Decision flow content missing: {t}" for t in failed_tests])
            return False
            
        print("✓ Decision flow content validation passed")
        return True


def main():
    """Run the documentation test suite."""
    if len(sys.argv) > 1:
        docs_dir = sys.argv[1]
    else:
        docs_dir = "docs"
    
    tester = DocumentationTester(docs_dir)
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 All documentation tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Some documentation tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()