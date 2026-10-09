---
title: "Distributed Architecture"
author_profile: true
layout: single
---

[Documentation index]({{ site.baseurl }}{% link index.md %})

![Architecture]({{ '/images/architecture.png' | relative_url }})

**AI Hydra** components run independently and can be deployed across different machines.

### Example Deployment

This configuration demonstrates a fully distributed setup where simulation, routing, and visualization are physically separated across hosts. This example also uses custom ports. The *Hydra Client* is running locally, the *Hydra Router* is running on **bingo**, and the *Hydra Manager* is running on **islands**.

**1. Client**

```shell
# Client (local machine)
$ . hydra-venv/bin/activate
hydra-venv> ai-hydra-client --router-port 6757 --router-hb-port 6758 --server-pub-port 6760 --router-address bingo --server-address islands
```

**2. Router**

```shell
# Router (bingo)
$ . hydra-venv/bin/activate
hydra-venv> ai-hydra-router --port 6757 --hb-port 6758
```

**3. Server**

```shell
# Manager / Simulation (islands)
$ . hydra-venv/bin/activate
hydra-venv> ai-hydra-mgr --port 5759 --router-port 6757 --router-hb-port 6758 --router-address bingo
```
