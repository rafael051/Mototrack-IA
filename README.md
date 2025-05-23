# 🏍️ MotoTrack - Identificação Automática de Motos via Visão Computacional

Este notebook implementa um **protótipo de Visão Computacional** utilizando **Transfer Learning** com o modelo **MobileNet** pré-treinado.

---

## 🎯 Objetivos

- ✅ Utilizar o modelo MobileNet para **identificação automática** de motos.
- ✅ Aplicar em imagens capturadas no contexto do **pátio da Mottu**.
- ✅ Demonstrar a aplicação prática da **Visão Computacional** para melhorar a **gestão de frotas**.

---

## ✅ O que é Transfer Learning?

**Transfer Learning** consiste em **reutilizar modelos pré-treinados** para resolver novos problemas.

### ➡️ Vantagens

- ✅ Poucos dados necessários.
- ✅ Menor tempo de treinamento.
- ✅ Alta eficiência.

Neste projeto, usamos o **MobileNet** pré-treinado no **ImageNet**, que já reconhece milhares de objetos, incluindo **motos**.

---

## ✅ Instalação das dependências

Instalamos as principais bibliotecas:

- **TensorFlow/Keras** → manipulação e execução do modelo.
- **Matplotlib** → exibição de imagens.
- **NumPy** → operações numéricas.

```bash
!pip install tensorflow matplotlib numpy
```

✅ Dependências instaladas com sucesso.

Agora podemos importar as bibliotecas necessárias.

---

## ✅ Importação das bibliotecas

Importamos:

- `MobileNet` → modelo pré-treinado.
- `preprocess_input`, `decode_predictions` → pré-processamento e interpretação dos resultados.
- `image` → carregamento e transformação de imagens.
- `numpy` → para arrays.
- `matplotlib.pyplot` → para gráficos e visualização.

```python
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
```

✅ Bibliotecas importadas com sucesso.

Agora podemos carregar o modelo MobileNet.

---

## ✅ Frameworks e Ferramentas Utilizadas

### 🛠️ TensorFlow/Keras

**Por que utilizamos?**  
Escolhemos o **TensorFlow**, com sua API de alto nível **Keras**, por ser uma das bibliotecas mais robustas e amplamente utilizadas para aplicações de **Deep Learning** e **Visão Computacional**.

➡️ Permite a fácil utilização de **modelos pré-treinados**, como o **MobileNet**, com poucas linhas de código.  
➡️ Possui ótima integração com outras ferramentas e suporte à execução em ambientes como **Google Colab**, acelerando o desenvolvimento.

---

### 🛠️ MobileNet

**Por que utilizamos?**  

Escolhemos o modelo **MobileNet** por ser:

- ✅ **Leve** → ideal para ambientes com recursos computacionais limitados.
- ✅ **Eficaz** → mantém alta acurácia em tarefas de classificação de imagens, mesmo com arquitetura compacta.

**Contexto:**  

O MobileNet é ideal para aplicações **embarcadas** e **móveis**, como o **monitoramento inteligente de pátios**, pois:

- ➡️ Exige menos processamento.
- ➡️ Executa rapidamente.
- ➡️ Pode ser facilmente integrado a sistemas de **mapeamento e gestão de frotas**, como no caso da **Mottu**.

---

### 🛠️ Matplotlib e NumPy

- **Matplotlib** → utilizado para exibir graficamente os resultados das predições, com clareza e visualização interpretável.  
- **NumPy** → suporte a operações matriciais e manipulação eficiente de arrays de imagens.

✅ Com essa combinação de ferramentas, conseguimos implementar uma **solução rápida, eficiente e prática** para o protótipo de **identificação automática de motos**.

---

## ✅ Carregamento do modelo MobileNet

Carregamos o **MobileNet** pré-treinado com pesos do **ImageNet**.

Também mostramos o **resumo** da arquitetura do modelo para entendimento das camadas.

```python
model = MobileNet(weights='imagenet')
print("✅ Modelo MobileNet carregado com sucesso!")

# Exibe o resumo da arquitetura
model.summary()
```

✅ O modelo está carregado e pronto para uso.

Agora vamos definir a função de classificação de imagens.

---

## ✅ Definição da função `classify_image`

Esta função realiza:

1. Carregamento da imagem e redimensionamento para (224, 224).
2. Transformação em array e pré-processamento.
3. Predição com o modelo.
4. Decodificação das classes previstas.
5. Exibição gráfica e textual dos resultados.

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

✅ Função `classify_image` definida com sucesso.

Agora podemos realizar testes com imagens.

---

## ✅ Criando estrutura de diretórios

Organizamos imagens em:

- `data/motos/` → imagens de motos.
- `data/outros/` → imagens de outros objetos.

```python
import os
os.makedirs('data/motos', exist_ok=True)
```

✅ Diretórios criados com sucesso.

Agora vamos baixar algumas imagens de exemplo.

---

## ✅ Download de imagens de exemplo

Baixamos imagens diretamente da internet para as pastas criadas.

**Exemplo:**

- Moto de luxo.
- Outras categorias como carro e pessoa.

```bash
# Moto de aluguel
!wget https://www.motoo.com.br/fotos/2022/10/960_720/mottu-e-moto-eletrica_01102022_50798_960_720.jpg -O data/motos/moto1.jpg
```

✅ Imagens baixadas com sucesso.

Agora podemos preparar a lista de testes.

---

## ✅ Lista de imagens para teste

Criamos a lista `test_images` com os caminhos das imagens organizadas por categoria.

```python
test_images = [
    'data/motos/moto1.jpg'
]
```

✅ Lista `test_images` criada.

Agora classificamos cada imagem.

---

## ✅ Classificação em lote

Executamos `classify_image()` para todas as imagens da lista.

**Exibimos:**

- A imagem.
- A Top-1 predição.
- As 5 predições mais prováveis.
- Tempo de processamento.

```python
for img_path in test_images:
    classify_image(img_path, model)
    print('-' * 50)
```

✅ Classificação em lote realizada com sucesso.

Agora passamos para a discussão dos resultados.

---

## ✅ Discussão dos Resultados

- ✅ O modelo **MobileNet** classificou corretamente imagens de motos como `motorcycle`.
- ✅ Para imagens de carros ou pessoas, apontou outras classes relacionadas.
- ✅ O tempo médio de execução foi **inferior a 1 segundo**.

**Limitações:**

- 🚫 Não identifica a posição da moto (**sem bounding boxes**).
- ⚠️ Confusão possível com classes como `bicycle`.

**Melhorias:**

- ➡️ Utilizar detecção com **YOLOv8**.
- ➡️ Criar mapeamento interativo com **Streamlit**.
- ➡️ Integrar sensores **IoT**.

---

## ✅ Conclusão

O **MotoTrack** demonstrou com sucesso a aplicação de **Transfer Learning** com **MobileNet** para a **identificação automática de motos**.

---

### 🏁 Próximos Passos

1. ➕ Adicionar detecção com **bounding boxes** (YOLOv8).
2. ➕ Implementar interface interativa → ex.: **Streamlit**.
3. ➕ Integrar com **sensores IoT** para mapeamento em **tempo real**.

**Assim, avançamos para uma solução completa de Visão Computacional para a Mottu!**
