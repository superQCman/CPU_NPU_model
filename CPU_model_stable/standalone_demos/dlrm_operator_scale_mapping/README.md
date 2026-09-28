# DLRM 算子规模映射独立 Demo

本目录是独立交付 demo，用于说明 DLRM 上层负载参数如何映射到每个算子的规模特征。

## 文件

| 文件 | 作用 |
|---|---|
| `dlrm_operator_scale_mapping_demo.py` | 自包含脚本，只依赖 Python 标准库 |
| `sample_config.json` | 示例 DLRM 上层配置 |
| `expected_outputs/` | 使用示例配置生成的参考输出 |

## 运行

```bash
python standalone_demos/dlrm_operator_scale_mapping/dlrm_operator_scale_mapping_demo.py \
  --config-json standalone_demos/dlrm_operator_scale_mapping/sample_config.json \
  --out-dir standalone_demos/dlrm_operator_scale_mapping/expected_outputs \
  --self-test
```

也可以直接传参数：

```bash
python standalone_demos/dlrm_operator_scale_mapping/dlrm_operator_scale_mapping_demo.py \
  --batch-size 1024 \
  --num-indices-per-lookup 1000 \
  --arch-embedding-size 25600 \
  --arch-mlp-bot 400-300-200 \
  --arch-mlp-top 1600-800-1 \
  --inter-threads 1 \
  --num-threads 3 \
  --out-dir outputs/static_mapping_demo \
  --self-test
```

## 输出

| 输出文件 | 含义 |
|---|---|
| `dlrm_static_op_shapes_full_graph_76.csv` | 完整 76 节点输入/输出 tensor shape |
| `dlrm_static_features_full_graph_76.csv` | 完整 76 节点规模特征 |
| `dlrm_static_model_input_51.csv` | 当前性能模型实际读取的 51 节点特征 |
| `dlrm_operator_scale_mapping_summary.csv` | 面向人工审阅的算子规模解释表 |
| `dlrm_operator_scale_mapping_summary.json` | 带 metadata 和字段解释的 JSON 版 summary |

## 边界

- 不读取 profile 目录、shape CSV、timeline CSV。
- 不加载 model bundle。
- 不预测 latency。
- 脚本中显式定义完整 76 节点结构，便于交付审阅。
