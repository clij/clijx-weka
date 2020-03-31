package net.haesleinhuepf.clijx.weka;

import ij.ImagePlus;
import ij.process.FloatProcessor;
import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij2.CLIJ2;
import net.imglib2.img.array.ArrayImgs;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.*;

public class CLIJxWeka2Test {
    @Test
    public void test() {
        float[] f_ground_truth = {2,2,2,2,2,1,1,1,1,1};
        float[] f_measure1 =     {0,1,2,3,4,5,6,7,8,9};
        float[] f_measure2 =     {2,3,2,3,2,3,2,3,2,3};

        CLIJ2 clijx = CLIJ2.getInstance();

        ClearCLBuffer ground_truth = clijx.push(ArrayImgs.floats(f_ground_truth, new long[]{10, 1, 1}));
        ClearCLBuffer measure1 = clijx.push(ArrayImgs.floats(f_measure1, new long[]{10, 1}));
        ClearCLBuffer measure2 = clijx.push(ArrayImgs.floats(f_measure2, new long[]{10, 1}));

        ClearCLBuffer featureStack = clijx.create(new long[]{10, 1, 2}, clijx.Float);

        // copy features into feature stack
        clijx.copySlice(measure1, featureStack, 0);
        clijx.copySlice(measure2, featureStack, 1);

        CLIJxWeka2 clijxweka = new CLIJxWeka2(clijx, featureStack, ground_truth);
        System.out.println(clijxweka.getClassifier());

        String modelFilename = "C:/structure/models/clijxwekatest.model";

        clijxweka.saveClassifier(modelFilename);

        // test trained model:
        {
            clijxweka.classification = null;
            ClearCLBuffer result = clijxweka.getClassification();
            ImagePlus converted = clijx.pull(result);
            FloatProcessor fp = (FloatProcessor)converted.getProcessor();
            System.out.println("Applied model: " + Arrays.toString((float[]) fp.getPixels()));
        }

        // Test with model from disc
        {
            CLIJxWeka2 clijxweka2 = new CLIJxWeka2(clijx, featureStack, modelFilename);
            ClearCLBuffer result = clijxweka2.getClassification();
            ImagePlus converted = clijx.pull(result);
            FloatProcessor fp = (FloatProcessor) converted.getProcessor();
            System.out.println("Applied model from dis: " + Arrays.toString((float[]) fp.getPixels()));
        }


        clijx.release(ground_truth);
        clijx.release(measure1);
        clijx.release(measure2);
        clijx.release(featureStack);

    }

}