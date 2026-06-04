FROM node:24-bookworm AS frontend-build
WORKDIR /workspace/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend ./
RUN npm run build

FROM gradle:8.14.3-jdk17 AS backend-build
WORKDIR /workspace/backend
COPY backend/settings.gradle backend/build.gradle ./
COPY backend/src ./src
COPY --from=frontend-build /workspace/frontend/dist ./src/main/resources/static
RUN gradle --no-daemon bootJar

FROM eclipse-temurin:17-jre-jammy
WORKDIR /app
ENV SERVER_PORT=8079
ENV DEBUG=false
COPY --from=backend-build /workspace/backend/build/libs/pingpong-web.jar /app/app.jar
EXPOSE 8079
ENTRYPOINT ["java", "-jar", "/app/app.jar"]
