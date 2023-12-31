# rwkv.c
Inference a RWKV model in pure C.

## Export checkpoint and vocabulary table.

To export those, you need install `pytorch` and `fire`.

Run `pip install fire` to install `fire`.

**export vocabulary table**:
```
python export.py export_vocab source target
```

**export checkpoint**
```
python export.py export_checkpoint source target
```

## To compile this project, just run

```
gcc -Ofast -fopenmp src/main.c src/simple_ndarray.c -lm -o run
```

Or you can replace `gcc` by `clang`.

## In process

This project is still in process. File `src/main.c` has some comment about input and output.

- [ ] The code is not concise enough; it requires refactoring.
- [ ] Implement int8 quantization.
- [ ] Execute the chat function directly.
