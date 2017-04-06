package com.jiantsing.test;
/**
 * 将mnist数据转成图片输出
 */
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MnistDataToImage {
	private static Logger log = LoggerFactory.getLogger(MnistDataToImage.class);
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		
		 DataSetIterator iter = new MyMnistDataSetIterator(1,10,false);
	     List<INDArray> featuresTrain = new ArrayList<>();
	     int imgno=1;
	     while(iter.hasNext()){
	       DataSet ds = iter.next();
	       featuresTrain.add(ds.getFeatureMatrix());
	       BufferedImage bi = new BufferedImage(28,28,BufferedImage.TYPE_BYTE_GRAY);
           for( int i=0; i<784; i++ ){
               bi.getRaster().setSample(i % 28, i / 28, 0, (int)(255*ds.getFeatureMatrix().getDouble(i)));
           }
           ImageIO.write(bi, "JPEG", new File("D:\\doc\\test\\"+imgno+".jpg"));
           imgno++;
//	       System.out.println(ds.getFeatureMatrix().size(1));
	     }

	}

}
