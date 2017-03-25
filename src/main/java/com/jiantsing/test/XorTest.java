package com.jiantsing.test;

import java.io.IOException;

import org.bytedeco.javacpp.opencv_ml.TrainData;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer.Builder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class XorTest {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		test();
        

	}
	public static void train(){
		
		// TODO Auto-generated method stub
				// list off input values, 4 training samples with data for 2
		        // input-neurons each
				//创建一个4*2的输入矩阵，初始值为0
		        INDArray input = Nd4j.zeros(4, 2); 

		        // correspondending list with expected output values, 4 training samples
		        // with data for 2 output-neurons each
		        //创建一个4*2的目标矩阵，初始值为0
		        INDArray labels = Nd4j.zeros(4, 2);

		        // create first dataset
		        // when first input=0 and second input=0
		        /*输入矩阵下标
		         	(0,0)  (0,1)      0    0  
					(1,0)  (1,1)      1    0
					(2,0)  (2,1)      0    1
					(3,0)  (3,1)      1    1
					
				  期望值
				    (0,0)  (0,1)      1    0 
					(1,0)  (1,1)      0    1
					(2,0)  (2,1)      0    1
					(3,0)  (3,1)      1    0
					
					 the labels (these should be binarized label matrices such that the specified label has a value of 1 in the desired column with the label)
					 列值为1的为期望的列的下标，因此列数就是分类数，列中为1的为期望列
		         */
		        input.putScalar(new int[]{0, 0}, 0);
		        input.putScalar(new int[]{0, 1}, 0);
		        // then the first output fires for false, and the second is 0 (see class
		        // comment)
		        labels.putScalar(new int[]{0, 0}, 1);
		        labels.putScalar(new int[]{0, 1}, 0);

		        // when first input=1 and second input=0
		        input.putScalar(new int[]{1, 0}, 1);
		        input.putScalar(new int[]{1, 1}, 0);
		        // then xor is true, therefore the second output neuron fires
		        labels.putScalar(new int[]{1, 0}, 0);
		        labels.putScalar(new int[]{1, 1}, 1);

		        // same as above
		        input.putScalar(new int[]{2, 0}, 0);
		        input.putScalar(new int[]{2, 1}, 1);
		        labels.putScalar(new int[]{2, 0}, 0);
		        labels.putScalar(new int[]{2, 1}, 1);

		        // when both inputs fire, xor is false again - the first output should
		        // fire
		        input.putScalar(new int[]{3, 0}, 1);
		        input.putScalar(new int[]{3, 1}, 1);
		        labels.putScalar(new int[]{3, 0}, 1);
		        labels.putScalar(new int[]{3, 1}, 0);

		        // create dataset object
		        DataSet ds = new DataSet(input, labels);

		        // Set up network configuration
		        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
		        // how often should the training set be run, we need something above
		        // 1000, or a higher learning-rate - found this values just by trial and
		        // error
		        builder.iterations(10000);//迭代次数，建议1000以上
		        // learning rate
		        builder.learningRate(0.1);//学习率
		        // fixed seed for the random generator, so any run of this program
		        // brings the same results - may not work if you do something like
		        // ds.shuffle()
		        builder.seed(123);//随机种子
		        // not applicable, this network is to small - but for bigger networks it
		        // can help that the network will not only recite the training data
		        builder.useDropConnect(false);
		        // a standard algorithm for moving on the error-plane, this one works
		        // best for me, LINE_GRADIENT_DESCENT or CONJUGATE_GRADIENT can do the
		        // job, too - it's an empirical value which one matches best to
		        // your problem
		        builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
		        // init the bias with 0 - empirical value, too
		        builder.biasInit(0);//偏置值？？？
		        // from "http://deeplearning4j.org/architecture": The networks can
		        // process the input more quickly and more accurately by ingesting
		        // minibatches 5-10 elements at a time in parallel.
		        // this example runs better without, because the dataset is smaller than
		        // the mini batch size
		        builder.miniBatch(false);//不适用并行计算，因为数据集规模小

		        // create a multilayer network with 2 layers (including the output
		        // layer, excluding the input payer)
		        ListBuilder listBuilder = builder.list();

		        DenseLayer.Builder hiddenLayerBuilder = new DenseLayer.Builder();//全连接层，隐藏层
		        // two input connections - simultaneously defines the number of input
		        // neurons, because it's the first non-input-layer
		        hiddenLayerBuilder.nIn(2);//两个输入值
		        // number of outgooing connections, nOut simultaneously defines the
		        // number of neurons in this layer
		        hiddenLayerBuilder.nOut(8);//输出为什么为4？？？，可以变
		        // put the output through the sigmoid function, to cap the output
		        // valuebetween 0 and 1
		        hiddenLayerBuilder.activation(Activation.SIGMOID);//阈值函数，s
		        // random initialize weights with values between 0 and 1
		        hiddenLayerBuilder.weightInit(WeightInit.DISTRIBUTION);
		        hiddenLayerBuilder.dist(new UniformDistribution(0, 1));

		        // build and set as layer 0
		        listBuilder.layer(0, hiddenLayerBuilder.build());

		        // MCXENT or NEGATIVELOGLIKELIHOOD (both are mathematically equivalent) work ok for this example - this
		        // function calculates the error-value (aka 'cost' or 'loss function value'), and quantifies the goodness
		        // or badness of a prediction, in a differentiable way
		        // For classification (with mutually exclusive classes, like here), use multiclass cross entropy, in conjunction
		        // with softmax activation function
		        Builder outputLayerBuilder = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD);
		        // must be the same amout as neurons in the layer before
		        outputLayerBuilder.nIn(8);//与隐藏层的out要相同？？？
		        // two neurons in this layer
		        outputLayerBuilder.nOut(2);//目标输出个数，与label矩阵一致
		        outputLayerBuilder.activation(Activation.SOFTMAX);
		        outputLayerBuilder.weightInit(WeightInit.DISTRIBUTION);
		        outputLayerBuilder.dist(new UniformDistribution(0, 1));//输出的上下限值
		        listBuilder.layer(1, outputLayerBuilder.build());

		        // no pretrain phase for this network
		        listBuilder.pretrain(false);

		        // seems to be mandatory
		        // according to agibsonccc: You typically only use that with
		        // pretrain(true) when you want to do pretrain/finetune without changing
		        // the previous layers finetuned weights that's for autoencoders and
		        // rbms
		        listBuilder.backprop(true);

		        // build and init the network, will check if everything is configured
		        // correct
		        MultiLayerConfiguration conf = listBuilder.build();
		        MultiLayerNetwork net = new MultiLayerNetwork(conf);
		        net.init();

		        // add an listener which outputs the error every 100 parameter updates
		        net.setListeners(new ScoreIterationListener(100));

		        // C&P from GravesLSTMCharModellingExample
		        // Print the number of parameters in the network (and for each layer)
		        Layer[] layers = net.getLayers();
		        int totalNumParams = 0;
		        for (int i = 0; i < layers.length; i++) {
		            int nParams = layers[i].numParams();//10,12=22数值是怎么来的？？？
		            System.out.println("Number of parameters in layer " + i + ": " + nParams);
		            totalNumParams += nParams;
		        }
		        System.out.println("Total number of network parameters: " + totalNumParams);

		        // here the actual learning takes place
		        net.fit(ds);

		        // create output for every training sample
		        INDArray output = net.output(ds.getFeatureMatrix());
		        System.out.println(output);//训练结果

		        // let Evaluation prints stats how often the right output had the
		        // highest value
		        Evaluation eval = new Evaluation(2);
		        
		        
		        
		        eval.eval(ds.getLabels(), output);
		        System.out.println(eval.stats());//打印归类结果
		        String filePath =  System.getProperty("user.dir") + "\\src\\main\\resources\\";
		        try {
					ModelSerializer.writeModel(net,filePath + "xor.net",true);
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
	}
	public static void test(){
		MultiLayerNetwork model;
		String filePath =  System.getProperty("user.dir") + "\\src\\main\\resources\\";
		try {
			model = ModelSerializer.restoreMultiLayerNetwork(filePath + "xor.net");
			System.out.println("测试结果：");
	        INDArray output2 = Nd4j.zeros(2, 2); 
	        output2.putScalar(new int[]{0, 0}, 0);
	        output2.putScalar(new int[]{0, 1}, 0);
	        output2.putScalar(new int[]{1, 0}, 1);
	        output2.putScalar(new int[]{1, 1}, 1);
	        
	        System.out.println(model.output(output2));//输出测试结果
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
	}

}
