# Development image for the Next.js frontend.
# NOTE: dev-focused (runs `next dev`); a production build/standalone
# output image can be added in a later phase.
FROM node:20-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

EXPOSE 3000

CMD ["npm", "run", "dev"]
