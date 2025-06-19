# Small GPT from Scratch Using ```PyTorch```

This is an implementation of GPT at very small scale with only 29.13 million parameters. 

Below are the different parameters used for developing this model.

<center>

| Hyper Parameter | Value |
|-----------------|--------|
| ```batch_size``` | 16|
| ```block_size``` | 1024*3//4|
| ```max_iter``` | 75000|
| ```eval_interval``` | 500|
| ```lr``` | 5e-4|
| ```eval_iters``` | 200|
| ```n_embed``` | 256|
| ```n_heads``` | 4|
| ```n_layers``` | 4|
| ```dropout``` | 0.2

</center>

This model is one modified version of model by Andrej Karpathy. 
Below are the differences.,

<center>

| Hyperparameter | Original Value                | Modified Value                |
|----------------|-------------------------------|-------------------------------|
| `batch_size`   | 64                            | 16                            |
| `block_size`   | 256                           | `1024 * 3 // 4` = 768         |
| `max_iters`    | 5000                          | 75000                         |
| `eval_interval`| 500                           | 500                           |
| `learning_rate` / `lr` | `3e-4`               | `5e-4`                        |
| `eval_iters`   | 200                           | 200                           |
| `n_embd` / `n_embed` | 384                   | 256                           |
| `n_head` / `n_heads` | 6                     | 4                             |
| `n_layer` / `n_layers` | 6                   | 4                             |
| `dropout`      | 0.2                           | 0.2                           |


</center>
# Sample Generation - Solution Maybe Wrong but still generating good language 😂😂!!!

![sample](/GPT/asset/sample.gif)

**For checkpoints, just ping me on LinkedIn or email.**