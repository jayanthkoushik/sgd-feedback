# SGD-Feedback

This is code for the paper
**[Improving Stochastic Gradient Descent with Feedback](https://arxiv.org/abs/1611.01505)**,
<br>
[Jayanth Koushik](https://www.cs.cmu.edu/~jkoushik)\*,
[Hiroaki Hayashi](https://www.cs.cmu.edu/~hiroakih)\*,
<br>
(\* equal contribution)
<br>

## Usage
All results from the paper, and more are in the `data` folder. For example `data/cnn/cifar10/eve.pkl` has the results for using Eve to optimize a CNN on CIFAR10. The pickle files contain the loss history and cross-validation parameters. Additionally, all results are visualized in a jupyter notebook `src/compare_opts.ipynb`. The fixed models used in the paper are in `src/models.py`. The models are implemented in Keras. The experiments can be run using `src/runexp.py`. Run this script with `--help` as an argument to see the interface. The code for the character language model is in `src/charnn.py`. It is implemented in Theano. A keras implementation of our algorithm Eve is in `src/eve.py`. A theano implementation is also available in `src/theano_utils.py`.

## Citation
If you find this code useful, please cite
```
@article{koushik2016improving,
  title={Improving Stochastic Gradient Descent with Feedback},
  author={Koushik, Jayanth and Hayashi, Hiroaki},
  journal={arXiv preprint arXiv:1611.01505},
  year={2016}
}
```
