# 🏍️ MotoTrack - Identificação Automática de Motos via Visão Computacional

Este notebook implementa um protótipo de **Visão Computacional** utilizando **Transfer Learning** com o modelo **MobileNet** pré-treinado.

---

## 🎯 Objetivos

- ✅ Utilizar o modelo **MobileNet** para identificação automática de motos.
- ✅ Aplicar em imagens capturadas no contexto do pátio da **Mottu**.
- ✅ Demonstrar a aplicação prática da **Visão Computacional** para melhorar a **gestão de frotas**.

---

## ❓ O que é Transfer Learning?

**Transfer Learning** consiste em reutilizar modelos pré-treinados para resolver novos problemas.

### ➡️ Vantagens:

- ✅ Poucos dados necessários.
- ✅ Menor tempo de treinamento.
- ✅ Alta eficiência.

Neste projeto, usamos o **MobileNet** pré-treinado no **ImageNet**, que já reconhece milhares de objetos, incluindo motos.

---

## 🛠️ Frameworks e Ferramentas Utilizadas

### 🔧 TensorFlow/Keras

- 💪 Uma das bibliotecas mais robustas para **Deep Learning** e **Visão Computacional**.
- 🚀 Permite fácil utilização de modelos pré-treinados como o **MobileNet**.
- ☁️ Integração ideal com ambientes como o **Google Colab**.

### 🔧 MobileNet

- ⚡ Leve: ideal para ambientes com recursos computacionais limitados.
- 🎯 Eficaz: mantém alta acurácia em tarefas de classificação de imagens, mesmo com arquitetura compacta.

**Contexto:**

O **MobileNet** é ideal para aplicações embarcadas e móveis, como o **monitoramento inteligente de pátios**, pois:

- 🏎️ Exige menos processamento.
- ⚡ Executa rapidamente.
- 🔗 Pode ser facilmente integrado a sistemas de mapeamento e gestão de frotas, como no caso da **Mottu**.

### 🔧 Matplotlib e NumPy

- 📊 **Matplotlib**: exibição gráfica clara e interpretável dos resultados.
- 🔢 **NumPy**: suporte a operações matriciais e manipulação eficiente de arrays de imagens.

---

## 📝 Como Usar este Notebook

### ✅ Passo 1: Baixe o notebook

- 🔗 Acesse este repositório.
- 📥 Faça o download do arquivo `.ipynb` deste notebook.

💡 **Dica:** clique em `Code` → `Download ZIP` ou baixe apenas o notebook desejado.

---

### ✅ Passo 2: Importe para o Google Drive

- 📂 Acesse o [Google Drive](https://drive.google.com).
- 🗂️ Crie uma pasta (opcional, recomendado para organização).
- 📤 Faça o upload do notebook `.ipynb` para o seu Google Drive.

---

### ✅ Passo 3: Abra no Google Colab

- 🖱️ No Google Drive, clique com o botão direito no notebook.
- ➡️ Selecione **"Abrir com"** → **"Google Colab"**.
- ✅ O notebook será carregado, pronto para execução.

---

## 🛠️ Passos principais dentro do notebook

### 📦 Instalação das dependências

```bash
!pip install tensorflow matplotlib numpy
```

---

### 📥 Importação das bibliotecas

```python
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
```

---

### ⚙️ Carregamento do modelo MobileNet

```python
model = MobileNet(weights='imagenet')
print("✅ Modelo MobileNet carregado com sucesso!")
```

---

### 🧩 Definição da função de classificação

```python
def classify_image(img_path, model):
    # Carrega e exibe forma original
    img = image.load_img(img_path, target_size=(224, 224))
    print(f"Shape da imagem original: {np.array(img).shape}")

    # Converte para array e pré-processa
    x = image.img_to_array(img)
    print(f"Shape após conversão em array: {x.shape}")

    x = np.expand_dims(x, axis=0)
    print(f"Shape após expansão: {x.shape}")

    x = preprocess_input(x)

    # Tempo de execução
    import time
    start = time.time()
    preds = model.predict(x)
    end = time.time()
    print(f"Tempo de processamento: {end - start:.2f} segundos")

    # Decodifica predições
    decoded_preds = decode_predictions(preds, top=5)[0]

    # Exibe imagem com Top-1 predição
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Top-1: {decoded_preds[0][1]} ({decoded_preds[0][2]*100:.2f}%)")
    plt.show()

    # Exibe todas as Top-5 predições
    print(f"Classificação para {img_path}:")
    for i, (imagenet_id, label, score) in enumerate(decoded_preds):
        print(f"{i+1}. {label}: {score * 100:.2f}%")
```

---

### 📂 Criando estrutura de diretórios

```python
import os
os.makedirs('data/motos', exist_ok=True)
```

---

### 📷 Download de imagens de exemplo

```bash
!wget https://www.motoo.com.br/fotos/2022/10/960_720/mottu-e-moto-eletrica_01102022_50798_960_720.jpg -O data/motos/moto1.jpg
```

---

### 📝 Lista de imagens para teste

```python
test_images = ['data/motos/moto1.jpg']
```

---

### 🚀 Classificação em lote

```python
for img_path in test_images:
    classify_image(img_path, model)
    print('-' * 50)
```

---

## ✅ Resultados esperados

- ✅ Identificação correta de imagens de motos como **"motorcycle"**.
- ✅ Exibição gráfica das imagens com a **Top-1 predição**.
- ✅ Apresentação textual das **5 predições mais prováveis**.
- ✅ Tempo médio de execução: inferior a **1 segundo**.

---

## 🧐 Discussão dos Resultados

### ✅ Pontos positivos:

- ⚡ Rápido e eficiente.
- 🎯 Classifica bem motos.

### ❗ Limitações:

- 🚫 Não identifica a posição exata da moto (**sem bounding boxes**).
- 🔄 Possível confusão com classes como **"bicycle"**.

### 🔮 Melhorias futuras:

- 🛠️ Utilizar detecção com **YOLOv8**.
- 🖥️ Criar interface interativa com **Streamlit**.
- 📡 Integrar sensores **IoT** para mapeamento em tempo real.

---

## ✅ Conclusão

O **MotoTrack** demonstrou com sucesso a aplicação de **Transfer Learning** com **MobileNet** para a **identificação automática de motos**.

### 🔜 Próximos passos:

- ➕ Adicionar detecção com **bounding boxes**.
- ➕ Implementar interface interativa.
- ➕ Integrar sensores **IoT**.

---

## 🚦 Resumindo

✅ **Baixe** → ✅ **Importe** → ✅ **Abra** → ✅ **Execute!**

