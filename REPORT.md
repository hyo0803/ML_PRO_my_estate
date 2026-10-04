# ДЗ 2 · CI/CD для estate-service

Пайплайн: [.github/workflows/ci.yml](.github/workflows/ci.yml), три job'ы: `tests` → `build` → `deploy`.

## Содержание отчета

| Пункт задания | Ссылка |
|---|---|
| Зеленый прогон со всеми тремя job | [#13 (merge PR в main)](https://github.com/hyo0803/ML_PRO_my_estate/actions/runs/37220656866), [#20](https://github.com/hyo0803/ML_PRO_my_estate/actions/runs/37225703059) |
| Пакет с образом (sha в теге) | [ghcr.io/hyo0803/estate-service](https://github.com/hyo0803/ML_PRO_my_estate/pkgs/container/estate-service), теги вида `sha-<commit sha>` |
| Свой интеграционный тест (строка 200 и строка 422) | [tests/test_integration.py](tests/test_integration.py) |
| Своя настройка в ConfigMap (видна снаружи) | [k8s/configmap.yaml](k8s/configmap.yaml) → `MODEL_PATH`, `LOG_LEVEL` отдаются в `/health`; smoke проверяет `model_path` |
| Свой smoke | шаг `smoke` в job `deploy`: `/health` + `/v1/predict` с [good.json](good.json), цена в диапазоне 1e5–1e9 ₽, строка в БД с кодом 200 |
| PR с красным и зеленым прогоном | [PR #1](https://github.com/hyo0803/ML_PRO_my_estate/pull/1): красный [#11](https://github.com/hyo0803/ML_PRO_my_estate/actions/runs/37220477110) → зеленый [#12](https://github.com/hyo0803/ML_PRO_my_estate/actions/runs/37220582707) |
| Поломка 1 - конфиг | красный [#14](https://github.com/hyo0803/ML_PRO_my_estate/actions/runs/37221127864) → зеленый [#15](https://github.com/hyo0803/ML_PRO_my_estate/actions/runs/37223194923) |
| Поломка 2 - секрет | красный [#16](https://github.com/hyo0803/ML_PRO_my_estate/actions/runs/37223464471) → зеленый [#17](https://github.com/hyo0803/ML_PRO_my_estate/actions/runs/37224286837) |
| Поломка 3 - ресурсы | красный [#18](https://github.com/hyo0803/ML_PRO_my_estate/actions/runs/37225049484) (манифест отклонен), красный [#19](https://github.com/hyo0803/ML_PRO_my_estate/actions/runs/37225259000) (под Pending) → зеленый [#20](https://github.com/hyo0803/ML_PRO_my_estate/actions/runs/37225703059) |

## Три поломки с диагнозом

Сначала меня смутило, что во всех трех поломках шаг «сервис» в логе выглядел почти одинаково: `rollout status` ждет 180 секунд и падает с `timed out waiting for the condition`. По одному этому сообщению понять причину невозможно. Различать их я научилась по шагу «диагностика»: в первую очередь смотрела на колонку `STATUS` у пода `estate-service-...`, а потом на то, есть ли у него логи.

**1. Конфиг: `MODEL_PATH: artifact/no_such_model.joblib`.** Покраснел job `deploy` на шаге «сервис». В диагностике под нового ReplicaSet был в статусе `CrashLoopBackOff` и успел перезапуститься 4 раза. Раз есть рестарты, значит, контейнер все-таки запускался, и у него должны быть логи. В `kubectl logs` действительно нашелся трейсбэк: `FileNotFoundError: [Errno 2] No such file or directory: 'artifact/no_such_model.joblib'`, а за ним `Application startup failed`. Падение происходит в `lifespan` на `joblib.load`, и в ошибке напечатан тот самый путь, который пришел из ConfigMap. Так что даже без diff по логу понятно, что модель ищут не там.

**2. Секрет: `secretRef: {name: estate-secret}`, а пайплайн создает `estate-secrets`.** Снаружи все выглядело так же: красный `deploy`, шаг «сервис», таймаут. А вот статус пода другой: `CreateContainerConfigError`, рестартов 0. Я попробовала найти логи, но `kubectl logs` ответил `container "api" ... is waiting to start`. Тут до меня дошло, что контейнер не запускался ни разу, а значит, дело не в коде приложения, а в том, из чего Kubernetes собирает контейнер. Ответ нашелся в выводе `describe`, в блоке `Environment Variables from`: там указан секрет `estate-secret`, а на шаге «секреты» создавался `estate-secrets`. Разница всего в одну букву, и глазами ее легко пропустить.

**3. Ресурсы: `requests.memory: 1000000000Mi`.** С этой поломкой с первого раза вышло не то, что я ожидала. В [#18](https://github.com/hyo0803/ML_PRO_my_estate/actions/runs/37225049484) я поменяла только `requests`, а `limits` остался `1Gi`. Kubernetes отказался принимать такой Deployment прямо на `kubectl apply`: `must be less than or equal to memory limit of 1Gi`. Под вообще не создался, и диагностика написала только `deployments.apps "estate-service" not found`. request не может быть больше limit и это проверяется еще до запуска чего-либо. Во второй попытке ([#19](https://github.com/hyo0803/ML_PRO_my_estate/actions/runs/37225259000)) я подняла и limit. Тогда манифест приняли, и поды повисли в `Pending`. У них не было ни IP, ни узла (`NODE <none>`), ни логов, ни рестартов. Раз под даже не назначен на узел, значит, его не смог разместить планировщик: на узле kind просто нет миллиарда Mb памяти.

## Семь вопросов

**1. Время build и кэш.** В первом прогоне с job `build`, [#9](https://github.com/hyo0803/ML_PRO_my_estate/actions/runs/37218097638), он шел 50 секунд, из них шаг `build-push` занял 37. Во втором, [#13](https://github.com/hyo0803/ML_PRO_my_estate/actions/runs/37220656866), уже 34 секунды, а шаг 16. Чтобы понять, откуда разница, я открыла лог сборки #13. Слой `COPY src/ src/` там собирался заново (1.2 с), потому что в этом коммите я меняла код. А в следующем слое `uv sync --frozen --no-dev` установился всего один пакет за 0.6 с, и это сам проект. Все зависимости (pandas, scikit-learn и т.д.) уже были в слое `RUN uv sync --frozen --no-dev --no-install-project`, и он взялся из кэша. Его ключ зависит только от `pyproject.toml` и `uv.lock`, которые я не трогала, и стоит он в Dockerfile до копирования кода. Еще нагляднее прогон [#15](https://github.com/hyo0803/ML_PRO_my_estate/actions/runs/37223194923): там менялся только `k8s/configmap.yaml`, образ целиком взялся из кэша, шаг занял 4 секунды, а весь job 19.

**2. ImagePullBackOff при зеленом прогоне.** Эти поды я заметила в диагностике #14, #16 и #19 и сначала решила, что это еще одна ошибка. Оказалось, у меня в `k8s/deployment.yaml` указан образ `estate-service:1.0`. Первый `kubectl apply -f k8s/` создает ReplicaSet именно с ним, а в kind такого образа нет, и скачать его неоткуда. Но следующей же командой `kubectl set image` ставит настоящий образ `ghcr.io/...:sha-...`, появляется новый ReplicaSet, и когда он готов, старые поды удаляются. `rollout status` ждет только новый ReplicaSet, поэтому прогон зеленый.

**3. Путь пароля.** Сначала я создала секрет `DB_PASSWORD` в Settings → Secrets and variables → Actions. В шаге «секреты» он через `${{ secrets.DB_PASSWORD }}` попадает в переменную окружения этого шага, причем в логе GitHub сам заменяет его на `***`. Дальше `kubectl create secret generic estate-secrets` кладет его в Secret кластера: отдельно `POSTGRES_PASSWORD` и внутри `DATABASE_URL`. Deployment подключает этот Secret через `envFrom.secretRef`, и в поде появляется переменная окружения, которую читает `pydantic-settings`. Postgres берет тот же пароль через `secretKeyRef`. В `configmap.yaml` его класть нельзя: этот файл лежит в публичном репозитории, а ConfigMap вообще не предназначен для секретов и показывается открытым текстом в `kubectl describe`.

**4. Без `needs: tests` у build.** Тогда `build` и `tests` запустятся параллельно, и `build` не будет ждать результата тестов. Например, я ломаю что-то в коде: тесты падают через 40 секунд, но `build` к этому моменту уже собрал образ и опубликовал его в GHCR с тегом `sha-...`. В реестре оказывается образ, на котором тесты красные, и его можно случайно задеплоить. А поскольку `deploy` зависит только от `build`, сломанный код выкатится в кластер вообще автоматически.

**5. На PR только тесты.** За это отвечает строка `if: github.ref == 'refs/heads/main'` у job `build`, а `deploy` зависит от `build` через `needs` и пропускается вместе с ним. В PR `github.ref` равен `refs/pull/1/merge`, а не `main`, поэтому на странице PR #1 у `build` и `deploy` стоит skipped (прогоны #10–#12). Смысл в том, что код из ветки еще никто не проверил, и ему рано публиковать образ и выкатываться. Это происходит только после слияния в `main`.

**6. `pg_advisory_xact_lock` в `init()`.** У меня в Deployment `replicas: 2`, то есть при выкате одновременно стартуют два пода сервиса, и каждый вызывает `db.init()` с `CREATE TABLE IF NOT EXISTS`. `IF NOT EXISTS` не спасает: если база пустая, оба процесса одновременно видят, что таблицы нет, оба начинают ее создавать, и второй падает с `UniqueViolation`. После этого под уходит в рестарт. Advisory lock в транзакции выстраивает их в очередь: второй ждет, пока первый создаст таблицу, и потом просто видит, что она уже есть. Реплики здесь — это поды моего сервиса, а не копии самой базы.

**7. Статусы в порядке жизни пода.** Мне помогло представить три поломки как три разных места, где под «застревает» на пути к запуску. Сначала `Pending` (#19): под создан, но планировщик не нашел ему узел, и дальше дело не пошло. Потом `CreateContainerConfigError` (#16): узел найден, kubelet собирает контейнер, но не находит секрет из `envFrom`, и контейнер даже не создается. И последний `CrashLoopBackOff` (#14): контейнер создан и запущен, но приложение падает при старте, и kubelet перезапускает его снова и снова, каждый раз ожидая чуть дольше. Поэтому только в последнем случае есть логи приложения.

## Журнал проблем

**Тег образа без sha.** Когда я добавляла job `build`, имя образа у меня было `ghcr.io/<owner>/estate-service/sha-XXX`. Я думала, что sha попадает в тег, но на самом деле после последнего `/` идет часть имени пакета. Тега не было вообще, поэтому Docker ставил `latest`, а каждый коммит создавал бы новый пакет. Я исправила имя на `ghcr.io/<owner>/estate-service:sha-XXX`. Именно двоеточие отделяет тег от имени.

**Поломка «ресурсы» не дала Pending с первого раза.** Текст ошибки: `requests: Invalid value: "1000000000Mi": must be less than or equal to memory limit of 1Gi`. В логе шага «сервис» я увидела, что Deployment отклонил сам API-сервер, а в диагностике пода вообще не было, тк request не может быть больше limit. Я подняла `limits.memory` до того же значения, и в [#19](https://github.com/hyo0803/ML_PRO_my_estate/actions/runs/37225259000) получился нужный `Pending`.

**Ответы 422 не попадали в таблицу логов.** По заданию запрос с мусором тоже должен оставлять строку в базе с кодом 422. Я разобралась, что FastAPI проверяет запрос через Pydantic и сам отвечает 422 еще до вызова моего обработчика `/v1/predict`, поэтому до кода записи в базу дело не доходит. Я добавила обработчик `RequestValidationError`, который записывает такую строку с `score = NULL` и `status_code = 422` и возвращает в ответе `request_id`, чтобы тест мог эту строку найти.

Звездочки не делала.
