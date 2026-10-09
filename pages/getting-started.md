---
title: "Getting Started"
author_profile: true
layout: single
---

[Documentation index]({{ site.baseurl }}{% link index.md %})

## Installation

```shell
$ python3 -m venv hydra-venv
$ . hydra-venv/bin/activate
hydra-venv> pip install ai-hydra
```

---

## Startup

Start each component in its own terminal:

### 1. Client

```shell
$ . hydra-venv/bin/activate
hydra-venv> ai-hydra-client
```

### 2. Router

```shell
$ . hydra-venv/bin/activate
hydra-venv> ai-hydra-router
```

### 3. Server

```shell
$ . hydra-venv/bin/activate
hydra-venv> ai-hydra-mgr
```

#### Workflow

After starting the three processes:

1. Click **Start** in the *HydraRouter*
2. Click **Handshake** in the *HydraClient*

This sends a `HANDSHAKE` message through the router to the server.

If the client cannot connect to the server, then nothing will happen.

If the server is reachable, then the **Start** button appears, and the
simulation settings will also appear and be editable.

To launch the simulation, click on the **Start** button. Upon starting, the
**Pause** and **Reset** buttons will appear. These allow the user to pause a
running simulation, or reset the simulation, which takes you back to the
*post-handshake* display.

---

## Shutdown

In this early release, the server can only be stopped by hitting `Control-C`
in the terminal where it's running. The client is stopped by hitting the
**Quit** key. The *HydraRouter* can be shut down by clicking the *Quit* button.

The shutdown process is not 100% clean, so you may be required to also hit
`Control-C` on the client to fully stop it.

---

## Turbo Mode

The client supports a **Turbo mode** that disables per-step updates. This removes rendering overhead and allows the simulation to run **15×+ faster**, making it ideal for rapid experimentation.
