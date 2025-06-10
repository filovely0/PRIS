# ПРОЕКТИРОВАНИЕ ИНФОРМАЦИОННЫХ СИСТЕМ

## ВЫПОЛНИЛИ:
Черномор Маргарита Дмитриевна,
Строганова Елизавета Ивановна,
Либерман Алиса Ивановна,
Бурдь Ульяна Олеговна.

## Лабораторная работа №1: K8s, Minicube

### Установка minicube

1. Установить докер
2. Установить зависимости minikube `` $ sudo apt install -y curl wget apt-transport-https ``
3. Установить minicube

`` $ curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64 ``

`` $ sudo install minikube-linux-amd64 /usr/local/bin/minikube ``

4. Установить kubectl

`` $ curl -LO https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl ``

5. Выдать разрешения

`` $ chmod +x kubectl ``

`` $ sudo mv kubectl /usr/local/bin/ ``


### Запуск проекта

### развернуть minicube

`` minikube start --driver=docker --cpus=2 --memory=4096 ``

`` minikube addons enable metrics-server ``

`` minikube addons enable metrics-server dashboard ``

в новом терминале запускаем minikube dashboard, переходим по ссылке

### создать docker образ приложения

создаем папку проекта с названием kube, переходим

`` npm init -y ``

`` npm install express ``

создаем dockerfile

`` eval $(minikube docker-env) ``

`` docker build -t kube:1.0 . ``

### создать deployment

установить кол-во подов на 3

создаем файл deployment.yaml

создаем файл service.yaml

создаем app.js

### применить конфигурацию:

`` kubectl apply -f deployment.yaml ``
`` kubectl apply -f service.yaml ``

### добавить Metrics Server 

`` kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml ``

проверка работы kubectl top pods

`` kubectl get pods -l app=kube ``

### настройка Horizontal Pod Autoscaler (HPA), чтобы масштабировать поды при увеличении нагрузки 

`` kubectl autoscale deployment kube --cpu-percent=50 --min=2 --max=5 ``

`` kubectl get hpa ``

### настройка Prometheus и Grafana через (helm install prometheus prometheus-community/kube-prometheus-stack). Настройка дашбордов в Grafana для мониторинга нагрузки (хотя бы один)

`` helm repo add prometheus-community https://prometheus-community.github.io/helm-charts ``

`` helm repo update ``

`` helm install prometheus prometheus-community/kube-prometheus-stack ``

`` kubectl port-forward service/prometheus-grafana 3000:80 ``


### Откройте Grafana в браузере: http://localhost:3000

создать дашборд, в queries   

``sum(rate(container_cpu_usage_seconds_total{namespace="default", pod=~"kube-.+"}[1m])) by (pod) ``

потом меняем на

``kube_deployment_status_replicas{deployment="my-node-app"} ``

заходим в minicume dashmoard показываем что работает

Видео с демонстрацией лр1: https://drive.google.com/file/d/1QYJSICUDNYe-cseV4yiEyIptK-G01R1M/view?usp=sharing

## Лабораторная работа №2: Разработка микросервисной архитектуры с использованием GraphQL

### Запуск проекта

Установить необходимые инструменты:
Python 3.10 или выше
PostgreSQL
MongoDB
Node.js и npm

### Создать базы данных:

users_db в PostgreSQL

products_db в PostgreSQL

orders_db создается автоматически в MongoDB при первом запросе

### Установить зависимости для микросервисов на Python:

`` pip install fastapi uvicorn strawberry-graphql[fastapi] sqlalchemy psycopg2 pymongo motor ``

### Установить зависимости для Apollo Gateway на Node.js:

`` cd gateway ``

`` npm install ``

### Запустить каждый микросервис в отдельных терминалах:

`` cd users_service ``

`` uvicorn main:app --port 8001 ``

`` cd products_service ``

`` uvicorn main:app --port 8002 ``

`` cd orders_service ``

`` uvicorn main:app --port 8003 ``

### Запустить GraphQL Gateway:

`` cd gateway ``

`` node index.js ``

### Открыть в браузере:

http://localhost:4000

### Теперь можно выполнять все запросы через GraphQL Playground или Postman.

Видео с демонстрацией лр2: https://drive.google.com/file/d/1p79ebLZLWWvOrsaOt2mRM42KChwMA8NV/view?usp=sharing 



## Лабораторная работа №3: Работа с Big Data

### Запуск проекта

### Склонировать репозиторий репозиторий:

``git clone https://github.com/ИМЯ_ПОЛЬЗОВАТЕЛЯ/big-data-lab.git ``

`` cd big-data-lab ``

### Установить зависимости:

`` pip install -r requirements.txt ``

### Запустить Flask-приложение:

`` python app.py ``

### Открыть в браузере:

http://127.0.0.1:5000/

### Как пользоваться

1. Загрузить CSV-файл через веб-интерфейс.
2. Просмотреть таблицу и выбрать 2 признака для обучения.
3.Нажать "Обучить модель" для запуска.
4. Посмотреть результаты: график фактических и предсказанных значений, а также метрику MSE.

Видео с демонстрацией лр3: https://drive.google.com/file/d/1BoY-08kG2zS9e1WSKFHg0q5Tem8EsQR2/view?usp=sharing
