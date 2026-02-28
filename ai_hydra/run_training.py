# ai_hydra/run_training.py

from ai_hydra.server.HydraMgr import HydraMgr


def main():
    mgr = HydraMgr()  # does NOT need to start MQ server for local training
    stats, run = mgr.train_n_episodes(
        n_episodes=200,
        batch_size=64,
        train_every=1,
        max_steps=2000,
        print_every=1,
    )
    print(run)


if __name__ == "__main__":
    main()
