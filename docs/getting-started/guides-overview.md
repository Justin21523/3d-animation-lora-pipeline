# Documentation Guides

## Overview

This directory contains comprehensive guides for using the 3D Animation LoRA Pipeline.

## Available Guides

### Training & Optimization

#### **[Hyperparameter Optimization Guide](./HYPERPARAMETER_OPTIMIZATION_GUIDE.md)** (English)
Complete explanation of automated hyperparameter optimization using Optuna and TPE algorithm:
- How TPE (Tree-structured Parzen Estimator) works
- Search space design and rationale
- Multi-objective evaluation strategy
- Convergence guarantees and best practices
- Monitoring and diagnostics
- **Use this when**: Setting up or understanding the hyperparameter optimization system

#### **[超參數優化指南](./HYPERPARAMETER_OPTIMIZATION_GUIDE_ZH.md)** (中文版)
上述英文指南的完整中文翻譯，包含：
- TPE 算法原理和運作方式
- 搜尋空間設計
- 多目標評估策略
- 如何確保找到最佳參數
- 監控和診斷方法
- **適用場景**：需要理解或設置超參數優化系統時

### Tool-Specific Guides

*(To be added as more guides are created)*

- Frame Extraction Guide
- Layered Segmentation Guide
- Character Clustering Guide
- LoRA Testing Guide
- Video Generation Guide

## Quick Start

### New Users

If you're new to the pipeline, start with these guides in order:

1. Setup & Installation (coming soon)
2. Frame Extraction Guide (coming soon)
3. Character Clustering Guide (coming soon)
4. LoRA Training Guide (coming soon)
5. **Hyperparameter Optimization Guide** (current)

### Experienced Users

For specific tasks, jump directly to:

- **Improving LoRA quality**: [Hyperparameter Optimization Guide](./HYPERPARAMETER_OPTIMIZATION_GUIDE.md)
- **Understanding optimization**: [超參數優化指南](./HYPERPARAMETER_OPTIMIZATION_GUIDE_ZH.md)

## File Organization

```
docs/
├── guides/                           # User guides and tutorials
│   ├── README.md                     # This file
│   ├── HYPERPARAMETER_OPTIMIZATION_GUIDE.md     # English version
│   ├── HYPERPARAMETER_OPTIMIZATION_GUIDE_ZH.md  # 中文版
│   └── (more guides to be added)
├── 3d_anime_specific/                # 3D animation-specific documentation
└── reference/                        # Technical references
```

## Contributing

When adding new guides:

1. **Place in appropriate directory**: `guides/` for user-facing tutorials
2. **Create both English and Chinese versions** if possible
3. **Update this README** to include the new guide
4. **Follow naming convention**:
   - English: `TOPIC_NAME_GUIDE.md`
   - Chinese: `TOPIC_NAME_GUIDE_ZH.md`

## Related Documentation

- **Project README**: `../../README.md` (if exists)
- **CLAUDE.md**: `../../CLAUDE.md` - Project instructions for AI assistants
- **Technical References**: `../reference/`
- **3D-Specific Docs**: `../3d_anime_specific/`

---

**Last Updated**: 2025-11-12
