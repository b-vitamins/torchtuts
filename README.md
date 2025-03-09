# PyTorch: A rigorous training program

This is a training program for those interested in mastering PyTorch.

## Themes

- **Philosophy & Architecture:** The deeper “design invariants” and engineering trade-offs behind PyTorch’s dynamic graph approach.
- **Foundational Internals:** Tensors, memory layouts, autograd, operators (C++/CUDA), dispatch system, frontends (Python, C++), and code transformations (FX).
- **Advanced Topics & Best Practices:** Distributed training, HPC optimization, pipeline orchestration (TorchX), production deployment (TorchScript, ONNX, ExecuTorch), domain libraries (vision, text, audio, RL, multimodal, recommenders), interpretability, and more.
- **Extensibility & Contribution:** Creating custom ops, bridging with performance libraries, reading/writing PyTorch C++ internals, and open-source contribution workflows.
- **Rigorous Hands-On Exercises:** Expect in-depth coding labs, heavy reading, codebase explorations, and problem sets in every lecture.

## Lecture Breakdown

### Lecture 1: PyTorch’s Philosophical Underpinnings & The Big Picture
- Historical evolution of deep learning frameworks.
- Dynamic vs. static computation graphs.
- PyTorch’s layered architecture.
- Source code structure and key directories.

### Lecture 2: Deep Dive into Tensors—Memory Format, Strides & Channels-Last
- Memory layout, stride mechanics, and their performance implications.
- Channels-last format for improved efficiency.
- Advanced tensor indexing and broadcasting.

### Lecture 3: Autograd Engine, In-Place Ops, and Gradient Mechanics
- Reverse-mode automatic differentiation.
- In-place operations and common pitfalls.
- Hooks for advanced debugging.

### Lecture 4: C++ Autograd Internals & Operator Dispatch Mechanisms
- ATen and c10 internals.
- How PyTorch dispatches operations to CPU/GPU kernels.
- Deep dive into operator registration and execution.

### Lecture 5: Building `nn.Module` from Scratch & PyTorch’s OOP Design
- Parameter tracking, module hierarchy, buffers vs. parameters.
- Implementing a custom `nn.Module`.
- Investigating complex layers.

### Lecture 6: Data Pipeline Architecture, TorchText/TorchVision/TorchAudio
- Efficient dataset loading and preprocessing.
- Memory optimization techniques.
- Integrating large-scale data sources.

### Lecture 7: Profiling & Performance Tuning—Holistic Trace Analysis, TensorBoard
- PyTorch profiling tools.
- Identifying and resolving performance bottlenecks.
- Optimizing data transfer and computation scheduling.

### Lecture 8: TorchScript, TorchX & Production Graph Abstractions
- Converting dynamic PyTorch models to static graphs.
- Deployment with TorchScript.
- TorchX for ML pipeline orchestration.

### Lecture 9: Custom Operators & CUDA Extensions—From Python to Kernel Launch
- Writing custom autograd functions.
- Creating C++ and CUDA extensions.
- Fused operations for HPC performance.

### Lecture 10: FX—Rewriting and Transforming Computation Graphs
- Torch.FX intermediate representation.
- Graph transformations for performance and debugging.
- Practical applications in quantization and optimizations.

### Lecture 11: Advanced Frontend APIs—Forward-Mode AD, C++ Frontend, Dynamic Parallelism
- Alternative differentiation methods.
- C++ Frontend for high-performance applications.
- Dynamic kernel launching from within PyTorch.

### Lecture 12: Model Optimization, Hyperparameter Search & Best Practices with Ax
- Hyperparameter tuning strategies.
- Efficient model checkpointing and reproducibility.
- Scaling training experiments.

### Lecture 13: Parallel & Distributed Training (DDP, RPC, Model Parallel)
- Multi-GPU and multi-node training.
- DistributedDataParallel (DDP) vs. Model Parallel strategies.
- Optimizing gradient communication.

### Lecture 14: Fully Sharded Data Parallel, Pipeline Parallelism & Large-Scale Systems
- Fully Sharded Data Parallel (FSDP) internals.
- Pipeline parallelism and micro-batch scheduling.
- Real-world case studies in large-scale training.

### Lecture 15: Edge Deployment with ExecuTorch & Mobile Inference
- Deploying PyTorch models on mobile devices.
- Optimizing for memory and latency constraints.
- Profiling and debugging mobile models.

### Lecture 16: ONNX & Production Inference
- Model export to ONNX format.
- ONNX Runtime optimizations.
- Enterprise deployment considerations.

### Lecture 17: Interpretability, Debugging & Explainability
- PyTorch Captum for model explainability.
- Activation visualization techniques.
- Debugging and anomaly detection.

### Lecture 18: TorchX & Pipeline Orchestration—Enterprise MLOps
- Multi-step ML pipelines with TorchX.
- Logging, model registry, and automated training workflows.

### Lecture 19: Multimodal Models & TorchMultimodal
- Combining text, vision, and audio modalities.
- Cross-attention mechanisms.
- Training and fine-tuning multimodal networks.

### Lecture 20-28:
- Advanced domain-specific deep dives in **computer vision, audio processing, NLP, recommendation systems, reinforcement learning, debugging, and open-source contribution**.

---
