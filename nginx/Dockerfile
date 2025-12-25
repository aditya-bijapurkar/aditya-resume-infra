FROM nginx:alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY certs /etc/ssl/cloudflare

EXPOSE 80
EXPOSE 443

CMD ["nginx", "-g", "daemon off;"]
