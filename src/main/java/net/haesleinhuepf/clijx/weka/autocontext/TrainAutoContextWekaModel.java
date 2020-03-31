package net.haesleinhuepf.clijx.weka.autocontext;

import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij.coremem.enums.NativeTypeEnum;
import net.haesleinhuepf.clij.macro.CLIJMacroPlugin;
import net.haesleinhuepf.clij.macro.CLIJOpenCLProcessor;
import net.haesleinhuepf.clij.macro.documentation.OffersDocumentation;
import net.haesleinhuepf.clij2.AbstractCLIJ2Plugin;
import net.haesleinhuepf.clij2.CLIJ2;
import net.haesleinhuepf.clij2.utilities.ProcessableInTiles;
import net.haesleinhuepf.clijx.weka.*;
import org.scijava.plugin.Plugin;

/**
 * Author: @haesleinhuepf
 *         March 2020
 */

@Plugin(type = CLIJMacroPlugin.class, name = "CLIJx_trainAutoContextWekaModel")
public class TrainAutoContextWekaModel extends AbstractCLIJ2Plugin implements CLIJMacroPlugin, CLIJOpenCLProcessor, OffersDocumentation, ProcessableInTiles {

    @Override
    public boolean executeCL() {
        ClearCLBuffer input = (ClearCLBuffer)( args[0]);
        ClearCLBuffer ground_truth = (ClearCLBuffer)(args[1]);
        String modelFilename =  (String)args[2];
        String featureDefinitions = (String)args[3];
        int numberOfAutoContextIterations = asInteger(args[4]);
        int numberOfTrees = asInteger(args[5]);
        int numberOfFeatures = asInteger(args[6]);
        int maxDepth = asInteger(args[7]);

        return trainAutoContextWekaModelWithOptions(getCLIJ2(), input, ground_truth, modelFilename, featureDefinitions, numberOfAutoContextIterations, numberOfTrees, numberOfFeatures, maxDepth);
    }

    @Override
    public String getParameterHelpText() {
        return "Image input, Image ground_truth, String model_filename, String feature_definitions, Number numberOfAutoContextIterations, Number numberOfTrees, Number numberOfFeatures, Number maxDepth";
    }

    public static boolean trainAutoContextWekaModelWithOptions(CLIJ2 clij2, ClearCLBuffer input2D, ClearCLBuffer srcGroundTruth2D, String saveModelFilename, String featureDefinitions, int numberOfAutoContextIterations, Integer numberOfTrees, Integer numberOfFeatures, Integer maxDepth) {

        ClearCLBuffer feature_stack = GenerateFeatureStack.generateFeatureStack(clij2, input2D, featureDefinitions);
        int numberOfGeneratedFeatures = (int) feature_stack.getDepth();

        // -------------------------------------------------------------------
        // train classifier, save it as .0.model and as .0.model.cl file
        String model_filename = saveModelFilename + ".0.model";
        CLIJxWeka2 clijxweka = TrainWekaModelWithOptions.trainWekaModelWithOptions(clij2, feature_stack, srcGroundTruth2D, model_filename, numberOfTrees, numberOfFeatures, maxDepth);

        // get probabilities from the first round
        int numberOfClasses = clijxweka.getNumberOfClasses();
        ClearCLBuffer probability_map = clij2.create(new long[]{input2D.getWidth(), input2D.getHeight(), numberOfClasses}, NativeTypeEnum.Float);

        System.out.println("numberOfAutoContextIterations " + numberOfAutoContextIterations);
        for (int iterationCount = 0; iterationCount < numberOfAutoContextIterations; iterationCount++) {
            System.out.println("i" + iterationCount + ": generate probability maps");
            GenerateWekaProbabilityMaps.generateWekaProbabilityMaps(clij2, feature_stack, probability_map, model_filename);
            clij2.show(probability_map,"probability map");

            ClearCLBuffer probability_slice = clij2.create(new long[]{input2D.getWidth(), input2D.getHeight()}, input2D.getNativeType());
            ClearCLBuffer class_feature_stack = clij2.create(new long[]{input2D.getWidth(), input2D.getHeight(), numberOfGeneratedFeatures}, input2D.getNativeType());
            ClearCLBuffer combined_feature_stack = null;
            for (int c = 0; c < numberOfClasses; c++) {
                clij2.copySlice(probability_map, probability_slice, c);

                System.out.println("i" + iterationCount + " c" + c + ": generate feature stack");
                GenerateFeatureStack.generateFeatureStack(clij2, probability_slice, class_feature_stack, featureDefinitions);

                if (combined_feature_stack != null) {
                    clij2.release(combined_feature_stack);
                }
                combined_feature_stack = clij2.create(new long[]{input2D.getWidth(), input2D.getHeight(), feature_stack.getDepth() + class_feature_stack.getDepth()});
                clij2.concatenateStacks(feature_stack, class_feature_stack, combined_feature_stack);

                ClearCLBuffer temp = combined_feature_stack;
                combined_feature_stack = feature_stack;
                feature_stack = temp;
            }
            clij2.release(probability_slice);
            clij2.release(class_feature_stack);
            clij2.release(combined_feature_stack);

            model_filename = saveModelFilename + "." + (iterationCount + 1) + ".model";
            System.out.println("i" + iterationCount + ": train");
            TrainWekaModelWithOptions.trainWekaModelWithOptions(clij2, feature_stack, srcGroundTruth2D, model_filename, numberOfTrees, numberOfFeatures, maxDepth);

        }

        clij2.release(probability_map);
        clij2.release(feature_stack);

        clij2.reportMemory();
        System.out.println("Bye.");

        return true;
    }

    @Override
    public String getDescription() {
        return "Trains a Weka model using functionality of Fijis Trainable Weka Segmentation plugin.\n" +
                "It generates a 3D feature stack as described in GenerateFeatureStack" +
                "and trains a Weka model. This model will be saved to disc.\n" +
                "The given groundTruth image is supposed to be a label map where pixels with value 1 represent class 1, " +
                "pixels with value 2 represent class 2 and so on. Pixels with value 0 will be ignored for training.\n\n" +
                "Default values for options are:\n" +
                "* trees = 200\n" +
                "* features = 2\n" +
                "* maxDepth = 0";
    }

    @Override
    public String getAvailableForDimensions() {
        return "2D";
    }

}
