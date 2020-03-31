package net.haesleinhuepf.clijx.weka;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij2.CLIJ2;

public class TestOnEMData {
    public static void main(String... args) {
        String imageFilename = "C:\\structure\\data\\unidesigner_groundtruth-drosophila-vnc\\stack1\\raw\\00.tif";
        String groundTruthFilename = "C:\\structure\\data\\unidesigner_groundtruth-drosophila-vnc\\stack1\\labels\\labels00000000.png";
        String feature_definition = "original gaussianblur=1 gaussianblur=5 gaussianblur=7 sobelofgaussian=1 sobelofgaussian=5 sobelofgaussian=7";
        String modelFilename = "C:\\structure\\code\\clij_weka_scripts\\em_segm\\test.model";

        new ImageJ();
        ImagePlus imp = IJ.openImage(imageFilename);
        IJ.run(imp, "32-bit", "");
        ImagePlus ground_truth = IJ.openImage(groundTruthFilename);
        IJ.run(ground_truth, "32-bit", "");

        CLIJ2 clijx = CLIJ2.getInstance();
        ClearCLBuffer input = clijx.push(imp);
        ClearCLBuffer input_ground_truth = clijx.push(ground_truth);

        ClearCLBuffer output = clijx.create(input);

        //clijx.stopWatch("");
        ClearCLBuffer featureStack = GenerateFeatureStack.generateFeatureStack(clijx, input, feature_definition);
        //clijx.stopWatch("generate feature stack");

        input_ground_truth = replaceIntensities(clijx, input_ground_truth);
        input_ground_truth = thinOutGroundTruth(clijx, input_ground_truth, 0.99f);
        //input_ground_truth = convertToFloat(clijx, input_ground_truth);

        clijx.show(input_ground_truth, "in gt");

        //clijx.stopWatch("thin ground truth");

        TrainWekaModelWithOptions.trainWekaModelWithOptions(clijx, featureStack, input_ground_truth, modelFilename, 200, 2, 5);
        //clijx.stopWatch("train");

        //CLIJxWeka weka = new CLIJxWeka(clijx, featureStack, modelFilename);

        ApplyWekaModel.applyWekaModel(clijx, featureStack, output, modelFilename);
        //clijx.stopWatch("CPU predict1");

        ApplyWekaModel.applyWekaModel(clijx, featureStack, output, modelFilename);
        //clijx.stopWatch("GPU predict2");
/*
        ApplyOCLWekaModel.applyOCLWekaModel(clijx, featureStack, output, modelFilename);
        //clijx.stopWatch("GPU predict1");

        ApplyOCLWekaModel.applyOCLWekaModel(clijx, featureStack, output, modelFilename);
        //clijx.stopWatch("GPU predict2");
*/
        clijx.show(output, "output");



    }


    private static ClearCLBuffer replaceIntensities(CLIJ2 clijx, ClearCLBuffer input_ground_truth) {
        ClearCLBuffer temp1 = input_ground_truth;
        ClearCLBuffer temp2 = clijx.create(input_ground_truth);

        int membrane = 1;
        int glia_extracellular = 2;
        int mitochondria = 3;
        int synapse = 4;
        int intracellular = 5;

        // 0   -> membrane | (0°)
        clijx.replaceIntensity(temp1, temp2, 0, membrane);
        // 32  -> membrane / (45°)
        clijx.replaceIntensity(temp2, temp1, 32, membrane);
        // 64  -> membrane - (90°)
        clijx.replaceIntensity(temp1, temp2, 64, membrane);
        // 96  -> membrane \ (135°)
        clijx.replaceIntensity(temp2, temp1, 96, membrane);
        // 128 -> membrane "junction"
        clijx.replaceIntensity(temp1, temp2, 128, membrane);
        // 159 -> glia/extracellular
        clijx.replaceIntensity(temp2, temp1, 159, glia_extracellular);
        // 191 -> mitochondria
        clijx.replaceIntensity(temp1, temp2, 191, mitochondria);
        // 223 -> synapse
        clijx.replaceIntensity(temp2, temp1, 223, synapse);
        // 255 -> intracellular
        clijx.replaceIntensity(temp1, temp2, 255, intracellular);

        clijx.copy(temp2, temp1);

        return input_ground_truth;
    }

    private static ClearCLBuffer thinOutGroundTruth(CLIJ2 clijx, ClearCLBuffer input_ground_truth, float amount) {
        ClearCLBuffer temp = clijx.create(input_ground_truth);
        ClearCLBuffer temp2 = clijx.create(input_ground_truth);

        clijx.setRandom(temp2, 0f, 1f);
        clijx.greaterConstant(temp2, temp, amount);
        clijx.mask(input_ground_truth, temp, temp2);
        clijx.copy(temp2, input_ground_truth);

        clijx.release(temp);
        clijx.release(temp2);

        return input_ground_truth;
    }


}
