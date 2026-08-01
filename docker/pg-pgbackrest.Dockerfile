# Postgres 17 + pgvector + pgBackRest.
#
# Postgres runs archive_command itself, inside the container, so pgbackrest has
# to live in the image — it cannot be a host binary. The base is pinned by
# digest so the standby can be rebuilt on exactly the same Postgres build that
# the primary streams from.
FROM pgvector/pgvector@sha256:d2ef61f42ef767baa5a1475393303cc235bcd92febd9d7014eddb48b41f3bad0

RUN apt-get update \
    && apt-get install -y --no-install-recommends pgbackrest \
    && rm -rf /var/lib/apt/lists/*

# pgbackrest writes its lock and log directories as the postgres user.
RUN install -d -o postgres -g postgres /var/log/pgbackrest /var/lib/pgbackrest /var/spool/pgbackrest
