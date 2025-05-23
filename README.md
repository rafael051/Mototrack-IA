# 🛵 MotoTrack - Identificação Automática de Motos via Visão Computacional

---

## 🎯 Descrição do Projeto

O **MotoTrack** é um **protótipo funcional** que utiliza **Visão Computacional** com **Transfer Learning**, através do modelo **MobileNet**, pré-treinado no **ImageNet**, para realizar a **identificação automática de motos** em imagens.

Este projeto foi desenvolvido no contexto do **desafio da Mottu**, com o objetivo de **otimizar a gestão e o mapeamento do pátio de motos**, reduzindo processos manuais, aumentando a eficiência operacional e oferecendo maior precisão no controle logístico das motos.

---

## ✅ Objetivos

- Aplicar o modelo **MobileNet** para identificar automaticamente **motos** em imagens.
- Utilizar **imagens reais** que simulam o ambiente do **pátio da Mottu**.
- Demonstrar a **viabilidade técnica** da solução como **prova de conceito**.
- Evidenciar como a **Visão Computacional** pode **automatizar e otimizar** processos de monitoramento e controle de frotas.

---

## ✅ Tecnologias Utilizadas

| Tecnologia            | Finalidade                                                              |
|---------------------- |-------------------------------------------------------------------------|
| **Python**            | Linguagem principal utilizada no desenvolvimento do projeto             |
| **TensorFlow/Keras**  | Framework para carregar e executar o modelo MobileNet                   |
| **MobileNet**         | Modelo pré-treinado eficiente para tarefas de classificação de imagens  |
| **Matplotlib**        | Visualização gráfica e interpretação dos resultados obtidos             |
| **NumPy**             | Manipulação e processamento eficiente de arrays de imagens              |

---

## ✅ Por que MobileNet?

- **Leve** → ideal para ambientes com **recursos computacionais limitados**.
- **Eficiente** → oferece **boa acurácia** em tarefas de classificação de imagens.
- **Ágil** → processamento rápido, adequado para aplicações **em tempo real**.

**Escolhemos o MobileNet por ser leve e eficaz, ideal para aplicações embarcadas como o monitoramento inteligente de pátios**, sendo totalmente apropriado para o **desafio proposto pela Mottu**.

---

## ✅ Estrutura do Projeto

```plaintext
MotoTrack/
├── data/
│   └── motos/
│       ├── moto1.jpg
├── mototrack.ipynb
├── README.md
