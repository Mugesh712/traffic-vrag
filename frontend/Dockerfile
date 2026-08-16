# Frontend dev server (M15).
#
# Deliberately the Vite DEV server, not a production nginx build: this is a
# research demo that gets iterated on, and `npm run dev` gives hot reload
# against a bind-mounted source tree. A production image would instead
# `npm run build` and serve dist/ from nginx -- noted so the choice reads as
# intentional rather than unfinished.
FROM node:22-slim

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

EXPOSE 5173
# --host binds 0.0.0.0 so the port is reachable from outside the container;
# Vite otherwise listens on localhost only and the mapped port appears dead.
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
