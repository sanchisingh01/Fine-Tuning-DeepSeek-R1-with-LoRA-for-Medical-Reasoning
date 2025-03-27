# Fine-Tuning-DeepSeek-R1-with-LoRA-for-Medical-Reasoning
# Fine-Tuning DeepSeek-R1-Distill-Llama-8B on Medical Chain-of-Thought Dataset  

🚀 **Enhancing medical reasoning capabilities using LoRA and parameter-efficient fine-tuning.**  

## 📌 Project Overview  

This project fine-tunes **DeepSeek-R1-Distill-Llama-8B** using the **Medical Chain-of-Thought Dataset** from Hugging Face. The objective is to improve structured reasoning in medical AI applications while optimizing computational efficiency using **Low-Rank Adaptation (LoRA)**.  

### **🔹 Model:**  
- `DeepSeek-R1-Distill-Llama-8B`  
- Fine-tuned with **LoRA (Low-Rank Adaptation)** for efficient training  

### **🔹 Dataset:**  
- **Medical Chain-of-Thought Dataset** (Hugging Face)  
- Focused on structured medical reasoning  

### **🔹 Techniques & Libraries:**  
- **LoRA (Low-Rank Adaptation)** for lightweight fine-tuning  
- **Hugging Face Transformers** for model implementation  
- **Unsloth** for optimized training and inference  
- **PEFT (Parameter-Efficient Fine-Tuning)** for selective training  
- **TRL (Transformers Reinforcement Learning)** for supervised fine-tuning  
- **PyTorch** as the deep learning framework  

---

## ⚙️ **Methodology**  

### 📖 **1. Data Preparation**  
✔ Load and preprocess the **Medical Chain-of-Thought Dataset** to ensure compatibility with the model.  

### 🎛 **2. Model Setup**  
✔ Initialize `DeepSeek-R1-Distill-Llama-8B` and apply **LoRA** to reduce training overhead.  

### 🏋️‍♂️ **3. Training Configuration**  
✔ Define hyperparameters using `TrainingArguments` and `SFTTrainer` from `trl`:  
  - **Batch Size:** `2 per device`  
  - **Gradient Accumulation Steps:** `4`  
  - **Learning Rate:** `2e-4`  
  - **Precision:** `fp16` or `bf16` (based on hardware)  
  - **Optimizer:** `adamw_8bit`  

### 🔥 **4. Training Execution**  
✔ Train the model with optimized parameters while tracking performance using **Weights & Biases (wandb)**.  

### 📊 **5. Evaluation & Results**  
✔ Measure improvements in **medical reasoning accuracy** and compare against baseline models.  
✔ The fine-tuned model demonstrates improved reasoning capabilities in medical contexts.  

## 🔍 Understanding LoRA (Low-Rank Adaptation)  

### What is LoRA?  
**Low-Rank Adaptation (LoRA)** is a **parameter-efficient fine-tuning (PEFT) technique** that reduces the number of trainable parameters in large language models (LLMs). Instead of modifying all model weights, LoRA **injects small low-rank matrices** into specific layers, enabling efficient fine-tuning while **preserving the original pre-trained model**.  

### Why Use LoRA for Fine-Tuning?  
Fine-tuning **large models like DeepSeek-R1-Distill-Llama-8B** can be **computationally expensive** due to their billions of parameters. **LoRA addresses this by:**  
✔ **Reducing memory consumption** – LoRA modifies only a fraction of the parameters, avoiding full model updates.  
✔ **Speeding up training** – Since fewer parameters are updated, training is faster and requires **less GPU memory**.  
✔ **Retaining general knowledge** – The original model weights remain frozen, ensuring the fine-tuned model retains **pre-existing knowledge** while adapting to new tasks.  

### How LoRA Works  
1️⃣ **Freeze the original model weights** to retain knowledge.  
2️⃣ **Inject small low-rank matrices (A & B) into transformer layers**.  
3️⃣ **Only train these LoRA parameters**, drastically reducing trainable parameters.  
4️⃣ During inference, **combine LoRA layers with base model weights** for predictions.  

### Why LoRA for This Project?  
In this project, we used **LoRA** to fine-tune **DeepSeek-R1-Distill-Llama-8B** on the **Medical Chain-of-Thought Dataset** for the following reasons:  
✅ **Memory Efficiency** – Standard fine-tuning of an 8B model requires **high-end GPUs**, but LoRA enables **training on consumer GPUs**.  
✅ **Maintains Model Integrity** – The pre-trained model's **reasoning ability remains intact**, with only **task-specific improvements** made.  
✅ **Faster Training & Lower Costs** – LoRA **reduces hardware requirements**, allowing efficient training without sacrificing performance.  

---

### 🛠️ LoRA Implementation in Code  
```python
from peft import LoraConfig, get_peft_model

# Define LoRA configuration
lora_config = LoraConfig(
    r=8,  # Low-rank dimension
    lora_alpha=32,  
    lora_dropout=0.05,  
    bias="none",
    target_modules=["q_proj", "v_proj"]  # Apply LoRA to attention layers
)

# Apply LoRA to pre-trained model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Verify LoRA parameter reduction

