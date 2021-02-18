# Sudoku
#### 运行SudokuModel里面的`Solve.py`文件即可识别图片并求解数独返回答案，可以通过前几行的`model_path`和`image_path`来选择使用的模型和要解答的图片

## DataProcess
里面包含三个处理数据的python文件（用来统一MNIST和EI339-CN dataset sjtu数据集的格式）:`data_process.py`, `data_process_3.py`, `data_process2.py`
## NetModel
里面包含四个用Keras实现的神经网络模型的构建和训练过程(`OriginalNet.py`, `ShallowNet.py`, `DeepNet.py`, `AlexNet.py`)，和一个numpy搭建的LeNet网络及其训练过程和训练结果(`LeNet5`)
## SudokuModel
`model_parameters`: 包含各个神经网络训练出来的参数，以\(^o^)/~
文件的形式保存
`Sudoku.py`:Sudoku类的建立，求解和纠错功能的实现
`Utilities.py`:包含find_puzzle()函数和extract_digit()函数来从一张图片中找到数独和从图片中识别数字
`Solve.py`:求解过程，修改前几行代码(`model_path`, `image_path`)然后运行即可自动识别并求解