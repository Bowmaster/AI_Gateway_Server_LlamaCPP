# Caddy with a DNS-01 provider plugin (required for wildcard certs without :80).
# Swap caddy-dns/cloudflare for your DNS provider's module if not Cloudflare.
FROM caddy:2-builder AS builder
RUN xcaddy build --with github.com/caddy-dns/cloudflare

FROM caddy:2
COPY --from=builder /usr/bin/caddy /usr/bin/caddy
