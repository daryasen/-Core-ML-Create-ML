# CoreMLDemo — ДЗ1 Core ML & Create ML

Демо-приложение загружает MobileNetV2 из официального PyTorch Hub, конвертирует модель в формат Core ML с помощью `coremltools` и классифицирует выбранное фото через Vision (`VNCoreMLRequest`). Все вычисления выполняются локально.

## Готовый запуск

1. Откройте `CoreMLDemo.xcodeproj` в Xcode.
2. Выберите симулятор iPhone 17 Pro.
3. Нажмите Run (`⌘R`).
4. На стартовом экране автоматически классифицируется встроенное тестовое фото собаки. Нажмите «Выбрать другое», чтобы проверить своё изображение и посмотреть Top‑3 классов.

## Повторная конвертация

Нужны Xcode 15+, `uv` и `xcodegen`. Из корня проекта:

```bash
chmod +x scripts/setup_and_convert.sh
./scripts/setup_and_convert.sh
```

Скрипт создаёт изолированное окружение Python 3.11, устанавливает `coremltools`, PyTorch и torchvision, загружает MobileNetV2 из `pytorch/vision`, конвертирует её в `CoreMLDemo/MobileNetV2.mlpackage` и генерирует Xcode-проект.

## Тесты

В Xcode нажмите `⌘U` или выполните:

```bash
xcodebuild test -project CoreMLDemo.xcodeproj -scheme CoreMLDemo -destination 'platform=iOS Simulator,name=iPhone 17 Pro'
```

Тесты проверяют загрузку сконвертированной модели и настоящий инференс встроенной фотографии с корректным диапазоном вероятностей.

## Соответствие ТЗ

- модель получена через PyTorch Hub;
- `coremltools` установлен в локальное Python-окружение;
- модель сконвертирована в Core ML (`.mlpackage`, современный эквивалент `.mlmodel`);
- модель интегрирована в нативное SwiftUI-приложение;
- инференс выполняется через Vision и `VNCoreMLRequest`;
- пользователь выбирает изображение из галереи и получает Top‑3 результата.
