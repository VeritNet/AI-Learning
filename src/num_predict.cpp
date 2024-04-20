#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>

using namespace std;

//随机生成-0.5至0.5数作为每个初始权重和偏置
std::vector<double> generateVector(int length) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-0.5, 0.5);
    std::vector<double> result(length);
    for (int i = 0; i <= length - 1; i++) {
        result[i] = dis(gen);
    }
    return result;
}

//将神经网络模型权重和偏置保存到文件
void saveNetwork(const std::vector<std::vector<std::vector<std::vector<double>>>>& tensor, const std::string& filename) {
	std::ofstream out(filename, std::ios::binary);
	for (const auto& three_dim : tensor) {
		for (const auto& two_dim : three_dim) {
			for (const auto& one_dim : two_dim) {
				for (const double& value : one_dim) {
					out.write(reinterpret_cast<const char*>(&value), sizeof(double));
				}
			}
		}
	}
	out.close();
}

void loadNetwork(const std::string& filename, std::vector<std::vector<std::vector<std::vector<double>>>>& tensor) {
	std::ifstream in(filename, std::ios::binary);
	for (auto& three_dim : tensor) {
		for (auto& two_dim : three_dim) {
			for (auto& one_dim : two_dim) {
				for (double& value : one_dim) {
					in.read(reinterpret_cast<char*>(&value), sizeof(double));
				}
			}
		}
	}
	in.close();
}

//获取灰度文本数据并解析为vector
vector<double> getData(std::string path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cout << "Failed" << std::endl;
        exit(1);
    }

    std::string content;
    std::getline(file, content);
    file.close();

    std::vector<double> numbers;
    std::stringstream ss(content);
    std::string token;

    while (std::getline(ss, token, ',')) {
        double number = std::stof(token);
        numbers.push_back(number);
    }

    return numbers;
}


//开发调用-打印向量
void pv(const std::vector<double>& vec) {
    std::cout << "[";
    for (std::size_t i = 0; i < vec.size(); ++i) {
        std::cout << std::fixed << std::setprecision(2) << vec[i];
        if (i != vec.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

//全局变量
std::vector<std::vector<std::vector<std::vector<double>>>> network;//模型权重和偏置
std::vector<std::vector<double>> networkn;//所有激活值
std::vector<std::vector<double>> networkb;//所有激活前的wx+b的值
double rate;//学习率
double aim;//目标损失值

//一个隐藏层神经元
vector<double> neuron(std::vector<double> w, std::vector<double> x, double b) {
    //激活函数-ELU
    auto elu = [](double x) {
        if (x >= 0) {
            return x;
        }
        else {
            return 1.0 * (exp(x) - 1);
        }
    };

    //求和∑wx+b
    auto sigma = [&w, &x, b]() {
        double sum = 0;
        for (int i = 0; i < w.size(); ++i) {
            sum += w[i] * x[i];
        }
        return sum + b;
    };

    double sum = sigma();//激活前的值

    return { sum,elu(sum) };
}

//一个输出层神经元
double sneuron(std::vector<double> w, std::vector<double> x, double b) {
    //求和∑wx+b
    auto sigma = [&w, &x, b]() {
        double sum = 0;
        for (int i = 0; i < x.size(); ++i) {
            sum += w[i] * x[i];
        }
        return sum + b;
    };

    return sigma();
}

//ELU激活函数的导数
double elu_derivative(double x) {
    if (x >= 0) {
        return 1;
    } else {
        return 1.0 * exp(x);
    }
}

//均方误差用于统计模型误差训练情况
double MSE(double out, double out_hat) {
    return (out - out_hat) * (out - out_hat);
}

//Softmax作为输出层激活函数
vector<double> softmax(std::vector<double> output, double sum) {
    std::vector<double> yhat(output.size(), 0);//输出层所有激活值
    for (int i = 0; i <= output.size()-1; i++) {
        yhat[i] += exp(output[i]) / sum;//计算每一个激活值
    }
    return yhat;
}


//预测-向前传播
vector<double> predict(vector<double> content) {
    //向前传播至隐藏层
    for (int m = 0; m <= networkn[0].size() - 1; m++) {
        auto r0 = neuron(network[0][m][0], content, network[0][m][1][0]);
        networkb[0][m] = r0[0];//未激活值
        networkn[0][m] = r0[1];//激活值
    }

    //向前传播至输出层
    for (int n = 0; n <= networkn[1].size() - 1; n++) {
        auto r1 = sneuron(network[1][n][0], networkn[0], network[1][n][1][0]);
        networkb[1][n] = r1;//未激活值
    }
    
    //softmax激活输出层
    double sum = 0;//Softmax的分母（每个激活值共用）
    for (int i = 0; i <= networkb[1].size()-1; i++) {
        sum += exp(networkb[1][i]);
    }
    networkn[1] = softmax(networkb[1], sum);//计算每个输出神经元激活值

    return networkn[1];
}


//训练-反向传播-随机梯度下降
double trainNet(vector<vector<double>> dt) {
    std::vector<double> out_hat = predict(dt[0]);//预测-前馈
    double MSError = 0;
    //计算误差值用MSE检查训练效果
    for (int l = 0; l <= out_hat.size() - 1; l++) {
        MSError += MSE(dt[1][l], out_hat[l]);
    }

    //为每个输出神经元分别计算学习率乘以损失值对输出层神经元未激活值的偏导，减轻后续计算量
    std::vector<double> rMEdN;
    for (int l = 0; l <= out_hat.size() - 1; l++) {
        rMEdN.push_back(rate * (out_hat[l] - dt[1][l]));//out_hat[l] - dt[1][l] 即Softmax+多分类交叉熵结合求导（使用交叉熵作为计算梯度时的损失函数，而使用MSE仅用于方便统计误差，不参与训练过程）
    }

    //更新输出层权重
    for (int p = 0; p <= networkn[1].size() - 1; p++) {//第p个输出神经元
        for (int q = 0; q <= network[1][p][0].size() - 1; q++) {//第q个权重
            network[1][p][0][q] -= rMEdN[p] * networkn[0][q];
        }
    }

    //更新输出层偏置
    for (int p = 0; p <= networkn[1].size() - 1; p++) {//第p个输出神经元
        network[1][p][1][0] -= rMEdN[p];
    }

    //更新隐藏层权重
    for (int p = 0; p <= networkn[0].size() - 1; p++) {//第p个隐藏神经元
        for (int q = 0; q <= network[0][p][0].size() - 1; q++) {//第q个权重
            //计算每个输出神经元的损失对此隐藏神经元的偏导，可以直接求和，也可以求平均（准确来说应该求和，但是这里就不再改变量名了，理解即可）
            double averagenN = 0;
            for (int s = 0; s <= network[1].size() - 1; s++) {
                averagenN += rMEdN[s] * network[1][s][0][p];//第s个输出神经元的梯度 乘以 连接着第s个输出神经元的权重
            }
            network[0][p][0][q] -= averagenN * elu_derivative(networkb[0][p]) * dt[0][q];
        }
    }

    //更新隐藏层偏置，同上
    for (int p = 0; p <= networkn[0].size() - 1; p++) {
        double averagenN = 0;
        for (int s = 0; s <= network[1].size() - 1; s++) {
            averagenN += rMEdN[s] * network[1][s][0][p];
        }
        network[0][p][1][0] -= averagenN * elu_derivative(networkb[0][p]);
    }

    return MSError;//返回统计误差
}

void train(vector<vector<vector<double>>> dt) {
    std::cout << "Gradient loss function: Cross Entropy" << std::endl;
    int i = 0;
    while (true) {
        i++;
        double err = 0;
        //梯度下降针对每个训练数据进行更新优化参数
        for (int c = 0; c <= dt.size() - 1; c++) {
            double preErr = trainNet(dt[c]);//梯度下降一次
            err += preErr;
        }
        
        if (i % 1 == 0) {
            rate *= 1.1;
        }
        if (i % 5 == 0) {
            rate *= 1.1;
        }

        //判断损失值是否满足要求即小于等于目标损失值
        if (err <= aim) {
            std::cout << "Training completed with err <= " << aim << " (" << err << ")" << std::endl;
            std::cout << ">>> finished " << dt.size() * i << " steps (" << i << " rounds) gradient descent in " << /*elapsed + */"ms <<<" << std::endl;
            break;
        }
        else {
            std::cout << "Round: " << i << "  Training: " << dt.size() * i << "  MSE: " << err << " rate: " << rate << std::endl;
        }
    }
}


int main() {
    networkn = {
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0}
    };
    networkb = {
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0}
    };
    
    /*训练完成后预测时使用以下代码
    loadNetwork("./num_predict.bin", network);
    //以下代码用于预测测试数据集
    for (int dt0 = 0; dt0 <= 9; dt0++) {
        cout << "----------" << dt0 << "----------" << endl;
        for (int dt1 = 0; dt1 <= 10; dt1++) {
            pv(predict(getData("./data/testing/" + to_string(dt0) + "/" + to_string(dt1) + ".txt")));
        }
    }
    //以下代码用于预测你手写的数字
    pv(predict(getData("./your_data_path.txt")));
    */

    //训练神经网络使用以下代码，预测时删掉以下代码
    cout << "1/3 Generate Vector" << endl;
    network = {
        {
            {generateVector(784), {0}},
            {generateVector(784), {0}},
            {generateVector(784), {0}},
            {generateVector(784), {0}},
            {generateVector(784), {0}},
            {generateVector(784), {0}},
            {generateVector(784), {0}},
            {generateVector(784), {0}},
            {generateVector(784), {0}},
            {generateVector(784), {0}},
            {generateVector(784), {0}},
            {generateVector(784), {0}},
            {generateVector(784), {0}},
            {generateVector(784), {0}},
            {generateVector(784), {0}},
            {generateVector(784), {0}},
            {generateVector(784), {0}},
            {generateVector(784), {0}},
            {generateVector(784), {0}},
            {generateVector(784), {0}},
            {generateVector(784), {0}},
            {generateVector(784), {0}},
            {generateVector(784), {0}},
            {generateVector(784), {0}},
            {generateVector(784), {0}},
            {generateVector(784), {0}},
            {generateVector(784), {0}},
            {generateVector(784), {0}},
            {generateVector(784), {0}},
            {generateVector(784), {0}},
        },
        {
            {generateVector(30), {0}},
            {generateVector(30), {0}},
            {generateVector(30), {0}},
            {generateVector(30), {0}},
            {generateVector(30), {0}},
            {generateVector(30), {0}},
            {generateVector(30), {0}},
            {generateVector(30), {0}},
            {generateVector(30), {0}},
            {generateVector(30), {0}}
        }
    };
    rate = 0.0015;//学习率
    aim = 1;//目标损失值
    std::vector<std::vector<std::vector<double>>> training_data;
    cout << "2/3 Load Training Data" << endl;
    for(int dt1=0;dt1<=100;dt1++){
        training_data.push_back({getData("./data/0/"+to_string(dt1)+".txt"), {1,0,0,0,0,0,0,0,0,0}});
        training_data.push_back({getData("./data/1/"+to_string(dt1)+".txt"), {0,1,0,0,0,0,0,0,0,0}});
        training_data.push_back({getData("./data/2/"+to_string(dt1)+".txt"), {0,0,1,0,0,0,0,0,0,0}});
        training_data.push_back({getData("./data/3/"+to_string(dt1)+".txt"), {0,0,0,1,0,0,0,0,0,0}});
        training_data.push_back({getData("./data/4/"+to_string(dt1)+".txt"), {0,0,0,0,1,0,0,0,0,0}});
        training_data.push_back({getData("./data/5/"+to_string(dt1)+".txt"), {0,0,0,0,0,1,0,0,0,0}});
        training_data.push_back({getData("./data/6/"+to_string(dt1)+".txt"), {0,0,0,0,0,0,1,0,0,0}});
        training_data.push_back({getData("./data/7/"+to_string(dt1)+".txt"), {0,0,0,0,0,0,0,1,0,0}});
        training_data.push_back({getData("./data/8/"+to_string(dt1)+".txt"), {0,0,0,0,0,0,0,0,1,0}});
        training_data.push_back({getData("./data/9/"+to_string(dt1)+".txt"), {0,0,0,0,0,0,0,0,0,1}});
    }
    cout << "3/3 Ready" << endl;
    train(training_data);
    saveNetwork(network, "./num_predict.bin");

    return 0;
}
