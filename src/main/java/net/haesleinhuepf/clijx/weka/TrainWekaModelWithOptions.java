package net.haesleinhuepf.clijx.weka;

import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij.macro.CLIJMacroPlugin;
import net.haesleinhuepf.clij.macro.CLIJOpenCLProcessor;
import net.haesleinhuepf.clij.macro.documentation.OffersDocumentation;
import net.haesleinhuepf.clij2.AbstractCLIJ2Plugin;
import net.haesleinhuepf.clij2.CLIJ2;
import net.haesleinhuepf.clij2.utilities.ProcessableInTiles;
import org.scijava.plugin.Plugin;

/**
 * Author: @haesleinhuepf
 *         March 2020
 */

@Plugin(type = CLIJMacroPlugin.class, name = "CLIJx_trainWekaModelWithOptions")
public class TrainWekaModelWithOptions extends AbstractCLIJ2Plugin implements CLIJMacroPlugin, CLIJOpenCLProcessor, OffersDocumentation, ProcessableInTiles {

    @Override
    public boolean executeCL() {
        trainWekaModelWithOptions(getCLIJ2(), (ClearCLBuffer)( args[0]), (ClearCLBuffer)(args[1]), (String)args[2], asInteger(args[3]), asInteger(args[4]), asInteger(args[5]));
        return true;
    }

    public static CLIJxWeka2 trainWekaModelWithOptions(CLIJ2 clij2, ClearCLBuffer srcFeatureStack3D, ClearCLBuffer srcGroundTruth2D, String saveModelFilename, Integer numberOfTrees, Integer numberOfFeatures, Integer maxDepth) {
        CLIJxWeka2 weka = new CLIJxWeka2(clij2, srcFeatureStack3D, srcGroundTruth2D);
        weka.setNumberOfTrees(numberOfTrees);
        weka.setNumberOfFeatures(numberOfFeatures);
        weka.setMaxDepth(maxDepth);
        System.out.println("Saved to " + saveModelFilename);
        weka.saveClassifier(saveModelFilename);
        return weka;
    }

    @Override
    public String getParameterHelpText() {
        return "Image featureStack3D, Image groundTruth2D, String saveModelFilename, Number trees, Number features, Number maxDepth";
    }

    @Override
    public String getDescription() {
        return "Trains a Weka model using functionality of Fijis Trainable Weka Segmentation plugin.\n" +
                "It takes a 3D feature stack (e.g. first plane original image, second plane blurred, third plane edge image)" +
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
