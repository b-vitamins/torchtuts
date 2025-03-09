# PyTorch: A rigorous training program

This is a training program for those interested in mastering PyTorch.

## Themes

1. **Philosophy & Architecture**: The deeper “design invariants” and engineering trade-offs behind PyTorch’s dynamic graph approach.  
2. **Foundational Internals**: Tensors, memory layouts, autograd, operators (C++/CUDA), dispatch system, frontends (Python, C++), and code transformations (FX).  
3. **Advanced Topics & Best Practices**: Distributed training, HPC optimization, pipeline orchestration (TorchX), production deployment (TorchScript, ONNX, ExecuTorch), domain libraries (vision, text, audio, RL, multimodal, recommenders), interpretability, and more.  
4. **Extensibility & Contribution**: Creating custom ops, bridging with performance libraries, reading/writing PyTorch C++ internals, and open-source contribution workflows.  
5. **Rigorous Hands-On Exercises**: Expect in-depth coding labs, heavy reading, codebase explorations, and SICP-level conceptual problem sets in every lecture.

---

## Lecture 1: PyTorch’s Philosophical Underpinnings & The Big Picture

**Learning Outcomes**
- Comprehend the historical and design motivations behind PyTorch’s “define-by-run” philosophy.  
- Situate PyTorch within the continuum of deep learning frameworks (Theano, Caffe, TensorFlow, JAX) and HPC ecosystems.  
- Appreciate the layered architecture: ATen, autograd, Python bindings, dispatch, TorchScript, etc.

**Outline**
1. **Historical Context**  
   - Evolution from Torch (Lua) → Theano → Static Graphs → PyTorch.  
   - The impetus for an imperative style.

2. **Dynamic vs. Static Graph**  
   - “Define-by-run” explained; trade-offs in performance, debugging, research iteration.

3. **Modular Architecture**  
   - ATen (C++ tensor library), autograd engine, the dispatcher, Python/C++ frontends.

4. **Reading the Source**  
   - A quick orientation: `torch` top-level modules, `csrc/`, `aten/`, `tests`.

---

## Lecture 2: Deep Dive into Tensors—Memory Format, Strides & Channels-Last

**Learning Outcomes**
- Understand how PyTorch Tensors are laid out in memory: strides, contiguity, views, “channels-last” format.  
- Learn how memory-format manipulations affect performance in HPC contexts (CPU vectorization, GPU coalescing).  
- Build an intuition for broadcasting, indexing, and debugging tricky shape manipulations.

**Outline**
1. **Strides & Contiguity**  
   - Memory layout, `stride()`, implications of slicing.  
   - The cost of non-contiguous Tensors.

2. **Channels-Last & Memory Format**  
   - Why channels-last can boost performance on modern hardware.  
   - Writing code to convert or check memory formats.

3. **Broadcasting & Advanced Indexing**  
   - Broadcasting rules, sub-dimension expansions.  
   - Weighted indexing, masked operations, indexing complexities.

4. **Performance Angle**  
   - GPU coalesced reads, CPU vectorization, pinned memory usage.

---

## Lecture 3: Autograd Engine, In-Place Ops, and Gradient Mechanics

**Learning Outcomes**
- Dive into PyTorch’s autograd: dynamic graph creation, backward traversal.  
- Understand the interplay of `requires_grad`, `.grad_fn`, leaf variables, and “AccumulateGrad.”  
- Diagnose subtle issues with in-place operations and “gradient anomalies.”

**Outline**
1. **Autograd 101**  
   - Reverse-mode AD, building the graph on-the-fly.  
   - Leaf vs. non-leaf Tensors, `.grad_fn`.

2. **In-Place Operations**  
   - Why `+=` can break the graph.  
   - The infamous “leaf Variable that requires grad is being used in an in-place operation.”

3. **Hooks**  
   - `register_hook` for advanced debugging or gradient manipulation.

4. **Forward-Mode AD**  
   - Current Beta features, Jacobian-vector products, HPC/Research contexts.

---

## Lecture 4: C++ Autograd Internals & Operator Dispatch Mechanisms

**Learning Outcomes**
- Examine how the autograd engine and operators are implemented in C++ (ATen, c10 dispatcher).  
- Understand how Python calls route to CPU/GPU kernels, the function-pointer dispatch table, specialized backends.  
- Gain the confidence to read and modify operator code in `aten/`.

**Outline**
1. **ATen & c10**  
   - Structure of CPU/CUDA/Other backends.  
   - Examples from `aten/src/ATen/native/` for standard ops (e.g., Add, MatMul).

2. **Dispatcher**  
   - How calls route from Python to C++ kernels (CPU, GPU, XLA).  
   - Understanding “registration” macros and naming conventions.

3. **C++ Autograd**  
   - Under-the-hood classes for gradient functions.  
   - The function registry for `backward()` calls.

4. **Practical Reading**  
   - Skim a simple op’s source (e.g., `torch.add`) to see forward + backward logic.

---

## Lecture 5: Building nn.Module from Scratch & PyTorch’s OOP Design

**Learning Outcomes**
- Understand how `nn.Module` organizes parameters, buffers, submodules, and the importance of Python’s OOP approach.  
- Write a minimal “module framework” that imitates PyTorch’s parameter registration.  
- Explore advanced `nn.*` layers—Conv, RNN, Transformers—and see the underlying patterns.

**Outline**
1. **nn.Module Internals**  
   - The constructor, `_parameters`, `_buffers`, `_modules` dictionaries.  
   - `named_parameters()`, `named_children()`, recursion in module trees.

2. **Parameter vs. Buffer**  
   - Why `register_buffer` differs from `register_parameter`.  
   - State that does not require gradients.

3. **Reimplementing Linear**  
   - Demonstrate a custom linear layer by registering weight/bias.  
   - Compare with `nn.Linear`.

4. **Examining Complex Modules**  
   - Overview of how conv/RNN layers build upon the same mechanism.

---

## Lecture 6: Data Pipeline Architecture, TorchText/TorchVision/TorchAudio

**Learning Outcomes**
- Master the deeper logic behind `Dataset`, `DataLoader`, and `Sampler` classes.  
- Understand CPU/GPU data flow, pinned memory, multiprocessing, distributed sampling.  
- Survey specialized libraries—`torchvision`, `torchaudio`, `torchtext`—for domain-specific data prep and augmentation.

**Outline**
1. **Dataset & DataLoader**  
   - Multiprocessing, how Python’s GIL is circumvented.  
   - `pin_memory=True` for faster GPU transfers.  
   - Advanced `collate_fn` usage and potential bottlenecks.

2. **Domain Libraries**  
   - TorchVision’s transforms, dataset classes.  
   - TorchAudio’s I/O, feature extraction, sample rates.  
   - TorchText’s tokenization, text transforms.

3. **Memory & HPC Tuning**  
   - Profiling data pipelines (Perf, nsys).  
   - Integrating large external data sources.

---

## Lecture 7: Profiling & Performance Tuning—Holistic Trace Analysis, TensorBoard

**Learning Outcomes**
- Gain a deep working knowledge of PyTorch’s profiling tools (`torch.profiler`, TensorBoard, etc.).  
- Interpret CPU/GPU timelines, kernel overlaps, identify optimization opportunities.  
- Investigate advanced pitfalls: sync points, CPU bottlenecks, kernel inefficiencies.

**Outline**
1. **PyTorch Profiler**  
   - Basic usage, capturing traces, exploring them in TensorBoard.  
   - `record_function` for custom scope analysis.

2. **Holistic Trace Analysis**  
   - Identifying hotspots or idle times.  
   - Operator-level timeline interpretation.

3. **Case Studies**  
   - CNN saturating GPU vs. data pipeline starving it.  
   - Mixed-precision training’s effect on timelines.

4. **Best Practices**  
   - Minimizing CPU-GPU synchronization, environment variables.

---

## Lecture 8: TorchScript, TorchX & Production Graph Abstractions

**Learning Outcomes**
- Understand how TorchScript translates dynamic graphs into a static IR for production.  
- Learn scripting vs. tracing, limitations, debugging techniques.  
- Explore TorchX for ML pipeline orchestration, multi-step workflows, enterprise best practices.

**Outline**
1. **TorchScript IR**  
   - Scripting vs. Tracing: underlying concepts, pros/cons.  
   - Inspecting TorchScript graphs (`graph`, `code` attributes).

2. **Deployment**  
   - Freedoms/constraints in production (C++ deployment, memory footprints, cross-platform).  
   - Example: using TorchScript in a microservice with libtorch.

3. **TorchX**  
   - Multi-step pipelines (data prep, training, evaluation, hyperparam search).  
   - Containerizing and scheduling on Kubernetes.

---

## Lecture 9: Custom Operators & CUDA Extensions—From Python to Kernel Launch

**Learning Outcomes**
- Master writing custom autograd Functions (Python-level) for specialized forward/backward.  
- Dive into C++/CUDA extensions, bridging Python calls to custom GPU kernels.  
- Explore fused operations for HPC performance.

**Outline**
1. **torch.autograd.Function**  
   - Overriding forward/backward, hooking into autograd.  
   - Use-cases: fused ops, specialized derivatives.

2. **C++/CUDA Extensions**  
   - Setup with setuptools or cmake, writing a minimal kernel, registering in Python.  
   - Memory safety, indexing, vectorization in CUDA.

3. **Practical Fusion**  
   - Combining activation + dropout in a single kernel for speed gains.

---

## Lecture 10: FX—Rewriting and Transforming Computation Graphs

**Learning Outcomes**
- Understand the Torch.FX IR, capturing Python model code as a graph for transformation.  
- Build passes that fuse modules, fold operations, or insert analysis.  
- Explore real-world uses—quantization (QAT), code generation, performance instrumentation.

**Outline**
1. **Overview of FX**  
   - Tracing a Python function into an IR.  
   - Graph representation (`GraphModule`, `Node` objects).

2. **Transformations**  
   - Folding Conv2D + BatchNorm.  
   - Rewriting activation functions or inlining submodules.

3. **Integration**  
   - Ties to `torch.compile`, AOTAutograd, Inductor backend.  
   - Instrumentation, custom graph rewriting, domain-specific optimizations.

---

## Lecture 11: Advanced Frontend APIs—Forward-Mode AD, C++ Frontend, Dynamic Parallelism

**Learning Outcomes**
- Explore beta features and lesser-known frontends: forward-mode AD, C++ Frontend for high-performance/embedded usage, dynamic parallelism in TorchScript.  
- Understand how these features serve HPC or specialized demands.

**Outline**
1. **Forward-Mode AD**  
   - Use-cases (Jacobian-vector products, high-order derivatives).  
   - Pros/cons vs. reverse-mode AD.

2. **C++ Frontend**  
   - Building a model in C++.  
   - Integrating with existing C++ HPC code or game engines.

3. **Dynamic Parallelism**  
   - GPU kernel launches from within kernels in TorchScript.  
   - Potential use cases for advanced GPU usage.

---

## Lecture 12: Model Optimization, Hyperparameter Search & Best Practices with Ax

**Learning Outcomes**
- Delve into advanced optimization strategies: hyperparameter search with Ax, iterative refinement.  
- Study best practices for large-scale experiments, checkpoint management, reproducibility.  
- Explore HPC cluster or cloud orchestration for large sweeps.

**Outline**
1. **Hyperparameter Optimization**  
   - Ax library (Bayesian optimization, bandit approaches).  
   - Integrating Ax with PyTorch training loops.

2. **Model Checkpointing & Versioning**  
   - Saving partial runs, continuing from checkpoints, Hydra configs.  
   - Ensuring reproducibility (random seeds, environment tracking).

3. **Scaling Up**  
   - Multi-GPU or distributed parameter sweeps.  
   - Cost analysis, HPC best practices.

---

## Lecture 13: Parallel & Distributed Training (DDP, RPC, Model Parallel)

**Learning Outcomes**
- Master fundamentals of distributed training: `torch.distributed`, spawn processes, ring-based allreduce.  
- Compare DataParallel, DistributedDataParallel (DDP), Model Parallel, Fully Sharded DataParallel (FSDP).  
- Explore RPC-based solutions (parameter servers, custom distributed ops).

**Outline**
1. **Distributed Overview**  
   - Initialization, “world size,” ranks, backends (Gloo, NCCL, MPI).  
   - Summaries of DDP, FSDP, Pipeline Parallelism.

2. **DDP Deep Dive**  
   - How gradient synchronization happens.  
   - Pitfalls (gradient divergence, batch size partitioning).

3. **Model Parallel**  
   - Splitting large models across GPUs.  
   - RPC usage for pipeline or parameter server approaches.

---

## Lecture 14: Fully Sharded Data Parallel, Pipeline Parallelism & Large-Scale Systems

**Learning Outcomes**
- Understand the mechanics of FSDP (sharding parameters, memory overhead, CPU offloading).  
- Explore pipeline parallelism for huge models (transformer-style).  
- Investigate real-case HPC workflows (GPT-scale training).

**Outline**
1. **FSDP Internals**  
   - Sharding strategies, memory usage, re-sharding parameters.  
   - Gains over standard DDP for large models.

2. **Pipeline Parallel**  
   - Splitting layers across multiple GPUs/hosts.  
   - Scheduling micro-batches, bubble overhead.

3. **Real HPC Scaling**  
   - Examples from large-scale training (GPT, BERT).  
   - HPC cluster specifics (Slurm, cluster scheduling).

---

## Lecture 15: Edge Deployment with ExecuTorch & Mobile Inference

**Learning Outcomes**
- Export models for ultra-lightweight Edge/IoT usage via ExecuTorch.  
- Understand iOS/Android builds, bridging C++ inference on mobile devices.  
- Profile edge models with the ExecuTorch SDK.

**Outline**
1. **ExecuTorch Overview**  
   - Why a specialized runtime? Constraints on edge devices (limited memory, CPU/GPU).  
   - Exporting a model to ExecuTorch.

2. **Mobile Builds**  
   - iOS/Android toolchains, minimal demo apps.  
   - Handling quantization, limited precision.

3. **Profiling**  
   - Using ExecuTorch SDK to measure latency, memory usage.  
   - Tuning batch sizes/input shapes for real-time constraints.

---

## Lecture 16: ONNX & Production Inference

**Learning Outcomes**
- Dive into ONNX for cross-framework model interoperability.  
- Understand how PyTorch → ONNX → ONNX Runtime fits into enterprise pipelines.  
- Explore dynamic shapes, custom operators, and performance considerations.

**Outline**
1. **ONNX Export**  
   - Tracing vs. Scripting pitfalls.  
   - Handling unsupported ops, shape mismatches.

2. **ONNX Runtime**  
   - Execution Providers (CPU, GPU, TensorRT).  
   - Production integration patterns.

3. **Case Study**  
   - Convert a moderately complex model (Transformer or custom RNN) to ONNX.  
   - Run in ONNX Runtime, measure speed.

---

## Lecture 17: Interpretability, Debugging & Explainability

**Learning Outcomes**
- Explore interpretability tooling in PyTorch (Captum or custom solutions).  
- Read/visualize saliency maps, integrated gradients, activation patterns.  
- Build robust debugging workflows that incorporate interpretability methods.

**Outline**
1. **Captum**  
   - APIs for integrated gradients, saliency, guided backprop.  
   - Visualization examples (CV, NLP).

2. **Debugging**  
   - Common training pathologies (vanishing/exploding gradients, data corruption).  
   - Use interpretability to detect data or architecture mistakes.

3. **Model Introspection**  
   - Tools for analyzing hidden representations.  
   - Distribution analysis (batch norm stats, layer stats).

---

## Lecture 18: TorchX & Pipeline Orchestration—Enterprise MLOps

**Learning Outcomes**
- Master advanced pipeline orchestration with TorchX (multi-step, distributed ML workflows).  
- Understand containerization, environment management, production-lifecycle best practices.  
- Explore logging, model registry, continuous training.

**Outline**
1. **TorchX Workflow**  
   - Job definition, scheduling on various backends (local, k8s, batch).  
   - Composable steps (data prep, training, evaluation, deployment).

2. **MLOps Integration**  
   - Model registry, versioning, continuous integration.  
   - Integrations with MLflow, Kubeflow, Weights & Biases.

3. **Advanced Patterns**  
   - Multi-tenant scheduling, large HPC cluster usage.  
   - Automated hyperparameter sweeps or ensemble training.

---

## Lecture 19: Multimodal Models & TorchMultimodal

**Learning Outcomes**
- Understand the fundamentals of multimodal ML (combining text, vision, audio, etc.).  
- Explore TorchMultimodal design (data structures, specialized layers like FLAVA).  
- Experiment with cross-attention, late-fusion, or other advanced patterns.

**Outline**
1. **Multimodal Basics**  
   - Motivation for fusing multiple modalities.  
   - Typical architectures for image-text tasks (tokenization vs. pixel processing).

2. **TorchMultimodal**  
   - FLAVA model architecture, fine-tuning details.  
   - Building custom encoders/decoders for new modalities.

3. **Practical Example**  
   - Train or fine-tune on image+text tasks.

---

## Lecture 20: Audio Domain Advanced—torchaudio & Speech Recognition

**Learning Outcomes**
- Gain advanced knowledge of torchaudio’s transforms, feature extraction, augmentation for speech.  
- Implement or fine-tune speech recognition with wav2vec2 or similar.  
- Understand time-frequency representations, sample rates, forced alignment.

**Outline**
1. **torchaudio Core**  
   - Waveform I/O, feature transforms (MelSpectrogram, MFCC), augmentation (speed perturbation).

2. **Speech Recognition Pipeline**  
   - wav2vec2 or RNN-based speech recognition.  
   - Handling variable-length inputs, beam search decoding.

3. **Forced Alignment & TTS**  
   - Aligning audio to text; TTS pipelines (Tacotron2).  
   - Potential expansions and custom modules.

---

## Lecture 21: Advanced NLP—Text Transformations, TorchText & Hugging Face Integrations

**Learning Outcomes**
- Master TorchText’s advanced pipelines: tokenization, vocab management, text transforms.  
- Integrate pretrained transformer architectures (BERT, GPT) via Hugging Face.  
- Explore subword tokenizers, custom dictionary merges, domain adaptation.

**Outline**
1. **TorchText**  
   - Building text pipelines, special tokens, multilingual corpora.  
   - Bucketing sequences, dynamic padding.

2. **Hugging Face Integration**  
   - Using pretrained BERT/GPT in PyTorch.  
   - Fine-tuning vs. feature extraction approaches.

3. **Advanced Topics**  
   - Custom subword segmentation, large corpora with distributed dataloaders.  
   - Domain adaptation for specialized tasks.

---

## Lecture 22: Transformers & Attention from First Principles

**Learning Outcomes**
- Achieve a thorough conceptual grounding in attention (scaled dot product, multi-head, cross-attention).  
- Implement a miniature Transformer from scratch (for NLP or vision).  
- Relate architecture decisions to HPC constraints (sequence lengths, memory usage).

**Outline**
1. **Attention Mechanism**  
   - Scaled dot product derivation.  
   - Queries, keys, values, multi-head splitting.

2. **Transformer Blocks**  
   - Layer normalization, feedforward sublayers, residual connections.  
   - Positional embeddings (sinusoidal vs. learned).

3. **Implementation**  
   - Mini-Transformer in PyTorch.  
   - Efficiency tips (attn masking, memory footprints).

4. **Vision Transformers**  
   - Adapting attention to images, patch embeddings.

---

## Lecture 23: Advanced Computer Vision—Detection, Segmentation, Video, TorchVision

**Learning Outcomes**
- Go beyond classification: object detection (Faster R-CNN), segmentation (Mask R-CNN), video tasks (optical flow, 3D convs).  
- Explore TorchVision’s model zoo, datasets for bounding boxes, segmentation masks.  
- Profile vision pipelines for HPC (heavy data loads, large batch sizes).

**Outline**
1. **Advanced Architectures**  
   - R-CNN family (Faster, Mask), YOLO-like approaches, segmentation basics.  
   - 3D convs for action recognition in video.

2. **TorchVision Tools**  
   - Pretrained detection models, bounding box transforms.  
   - Weighted sampling for class imbalance.

3. **Performance & HPC**  
   - Large image batch processing, memory constraints.  
   - Mixed-precision training for detection tasks.

---

## Lecture 24: Recommendation Systems with TorchRec

**Learning Outcomes**
- Investigate large-scale recommendation system pipelines, embedding tables, sharding strategies.  
- Understand TorchRec’s design (distributed embeddings, multi-GPU data flows, retrieval).  
- Implement or fine-tune a large embedding-based model, analyzing HPC issues.

**Outline**
1. **Recommender Fundamentals**  
   - Sparse embeddings (user/item IDs), factorization machines, DLRM.  
   - Large embedding tables, memory overhead.

2. **TorchRec**  
   - Sharding strategies (table-wise, row-wise, column-wise).  
   - Integration with distributed data parallel.

3. **Large-Scale Pipeline**  
   - Handling massive logs, negative sampling, evaluation metrics (CTR, recall).  
   - HPC environment for training huge embeddings.

---

## Lecture 25: Reinforcement Learning at Scale—TorchRL, Advanced Agents

**Learning Outcomes**
- Master advanced RL setups with TorchRL: environment registration, transforms, multi-step returns.  
- Explore algorithms beyond DQN (PPO, A2C, SAC) and advanced architectures for continuous control or discrete action spaces.  
- Scale RL with distributed training, parallel simulation.

**Outline**
1. **TorchRL**  
   - Building/wrapping custom gym-like environments.  
   - Environment transforms, replay buffers, parallel collectors.

2. **Advanced Algorithms**  
   - PPO/Actor-Critic pipelines, policy vs. value networks.  
   - TorchRL’s built-in trainers, distributed sampling.

3. **Scaling**  
   - Multi-GPU or multi-node RL training.  
   - HPC concerns: many short env calls, sync overhead.

---

## Lecture 26: Large-Scale Debugging & HPC Best Practices

**Learning Outcomes**
- Synthesize HPC-scale debugging: diagnosing GPU memory leaks, data race conditions, deadlocks.  
- Explore advanced logging frameworks, structured logs.  
- Understand HPC “best practices” for environment management, cluster usage.

**Outline**
1. **Common Failure Modes**  
   - Inf-nan errors, bounding them.  
   - Memory fragmentation, GPU OOM errors.

2. **Debugging Tooling**  
   - `nsys`, `nvprof`, GDB for PyTorch, memory leak detection in custom ops.

3. **HPC Environment**  
   - Docker vs. Singularity containers, cluster job scheduling.  
   - CPU affinity, GPU binding, environment modules.

---

## Lecture 27: Contributing to PyTorch Core & Emerging Features (Inductor, etc.)

**Learning Outcomes**
- Navigate PyTorch’s open-source workflow: reading PRs, tests, style guidelines, CI.  
- Explore new compilation technologies: Inductor, AOTAutograd, nvFuser.  
- Understand how to propose/land a feature or bug fix in the main repo.

**Outline**
1. **PyTorch Repo 201**  
   - Branching, pull requests, test suite (pytest, CI).  
   - Design docs, PyTorch RFC process.

2. **Inductor, nvFuser**  
   - Graph-lowering approach, kernel generation, synergy with FX.  
   - Beta features, how to experiment or debug them.

3. **Contribution Example**  
   - A hypothetical PR that adds or improves an op, including tests.

---

## Lecture 28: Final Projects, Synthesis, and Looking Beyond

**Learning Outcomes**
- Present final projects synthesizing the 14-week journey (custom ops, distributed training, domain specialties, interpretability, HPC, etc.).  
- Reflect on how PyTorch knowledge generalizes to other frameworks (TensorFlow eager mode, JAX).  
- Chart personal “next steps” for continued growth and open-source engagement.

**Outline**
1. **Final Project Presentations**  
   - Students showcase advanced RL, multimodal systems, HPC-scale training, custom operators, etc.

2. **Reflection**  
   - Compare experiences with dynamic graph frameworks; discuss partial compilation, AOT trends.

3. **Next Steps**  
   - Continued contribution, bridging frameworks, HPC developments.  
   - Encouragement to explore cutting-edge research code in PyTorch.

---
