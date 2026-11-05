# HTTPS for the UI on a LAN (optional)

`make ui` serves plain HTTP. On one laptop that is fine. When colleagues open the UI over a network, put TLS in front
of it. This is a compose profile (`caddy`), off by default: `make up` never starts it.

Caddy terminates HTTPS with a certificate from **its own local CA** (no internet, no public ACME) and proxies to
Streamlit, WebSockets included.

## Set up

1. In `.env` (optional; defaults in brackets): `RAG_LAN_HOST` [`localhost`] is the name people type, for example
   `rag.lan`; `RAG_HTTPS_PORT` [`8443`] is the published port (`443` works where nothing else uses it).
   The name must resolve on the clients (DNS, router, or `/etc/hosts`).
2. Start the UI: `make ui`. Keep Streamlit on the host.
3. Start the proxy: `make lan-tls`. Stop it with `make lan-tls-down`.
4. Install the CA on each device that opens the UI: `make lan-tls-ca` writes `data/caddy-root.crt` (a public
   certificate). Add it as a trusted root on those devices. Until then browsers warn.

Then open `https://<RAG_LAN_HOST>:<RAG_HTTPS_PORT>/`.

## How the proxy reaches Streamlit

The container reaches the host as `host.docker.internal` (`RAG_UPSTREAM`, default `host.docker.internal:8501`).

* **macOS / Windows (Docker Desktop)**: works with Streamlit bound to loopback. Set
  `STREAMLIT_SERVER_ADDRESS=127.0.0.1` so the plain-HTTP port is not also offered to the LAN. Checked on macOS
  with Docker Desktop: HTTPS, HTTP/2, a local-CA certificate, and a WebSocket upgrade through the proxy.
* **Linux**: `host.docker.internal` resolves to the Docker bridge gateway, so Streamlit must listen on an address the
  container can reach. Either run Streamlit on `0.0.0.0` and block port 8501 from the LAN with the host firewall, or
  run Caddy with host networking and set `RAG_UPSTREAM=127.0.0.1:8501`. This path is not tested here.

## What it does not do

* The CA key lives in the `caddy_data` volume. Anyone who can read it can issue certificates your devices trust:
  install the root only on devices you control, and delete the volume to retire the CA.
* It does not authenticate anyone: sign-in is still the application's (password, optional TOTP).
* The image is `caddy:2-alpine`, a moving tag. Pin a digest for a deployment you keep.
* For internet exposure use a public name and ACME instead of `tls internal`, and put the host behind a firewall.
