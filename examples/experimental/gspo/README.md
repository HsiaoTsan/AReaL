# Group Sequence Policy Optimization (GSPO)

This directory contains an example implementation of GSPO for training language models on the GSM8K mathematical reasoning task.

## What is GSPO?

GSPO (Group Sequence Policy Optimization) is a variant of PPO that replaces per-token probability ratio computation with sequence-level geometric mean of per-token probability ratios.

### Key Difference from Vanilla PPO

- **Vanilla PPO** (`importance_sampling_level: token`): Computes a separate importance sampling ratio for each token: `ratio[i] = exp(logprob[i] - old_logprob[i])`
- **GSPO** (`importance_sampling_level: sequence`): Computes one ratio per sequence as the geometric mean: `ratio = exp(mean(logprob - old_logprob))`

This sequence-level ratio is then applied uniformly to all tokens within each sequence, which can lead to more stable policy updates when optimizing for sequence-level rewards.

## Usage

To enable GSPO, set `importance_sampling_level: sequence` in the actor configuration:

```yaml
actor:
  importance_sampling_level: sequence  # 'token' for standard PPO, 'sequence' for GSPO
  # ... other configurations
```

## Running the Example

```bash
# Run GSPO training on GSM8K
python examples/experimental/gspo/gsm8k_gspo.py --config examples/experimental/gspo/gsm8k_gspo.yaml
```

## Configuration

The example configuration (`gsm8k_gspo.yaml`) includes:

- **importance_sampling_level: sequence** - Enables GSPO algorithm (sequence-level importance sampling)
- **group_size: 4** - Number of sequences per group
- **dynamic_sampling: true** - Filters out groups with identical rewards
- **use_decoupled_loss: true** - Uses decoupled PPO loss for off-policy correction
- **behav_imp_weight_cap: 5.0** - Caps behavioral importance weights

## Implementation Details

The GSPO implementation is located in `areal/utils/functional.py`. The algorithm:

1. Computes log probability ratios: `log_ratio = logprobs - proximal_logprobs`
2. Computes mean log ratio per sequence: `seq_log_ratio_mean = mean(log_ratio, dim=sequence_length)`
3. Applies geometric mean: `ratio = exp(seq_log_ratio_mean)`
4. Broadcasts the sequence-level ratio to all tokens in each sequence

## When to Use GSPO

GSPO may be beneficial when:
- Training with sequence-level rewards (e.g., task success/failure)
- Dealing with high variance in per-token gradients
- Optimizing for long-horizon tasks where token-level credit assignment is difficult

## Comparison with DAPO

While DAPO (Dynamic Adaptive Policy Optimization) focuses on dynamic sampling and reward filtering, GSPO focuses on the computation of importance sampling ratios. These approaches are complementary and can be used together.
