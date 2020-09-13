package net.haesleinhuepf.clijx.weka;

import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij.macro.CLIJMacroPlugin;
import net.haesleinhuepf.clij.macro.CLIJOpenCLProcessor;
import net.haesleinhuepf.clij.macro.documentation.OffersDocumentation;
import net.haesleinhuepf.clij2.AbstractCLIJ2Plugin;
import net.haesleinhuepf.clij2.CLIJ2;
import net.haesleinhuepf.clij2.utilities.IsCategorized;
import net.haesleinhuepf.clij2.utilities.ProcessableInTiles;
import org.scijava.plugin.Plugin;

/**
 * Author: @haesleinhuepf
 *         November 2019
 */

@Plugin(type = CLIJMacroPlugin.class, name = "CLIJx_trainWekaModel")
public class TrainWekaModel extends AbstractCLIJ2Plugin implements CLIJMacroPlugin, CLIJOpenCLProcessor, OffersDocumentation, ProcessableInTiles, IsCategorized {

    @Override
    public boolean executeCL() {
        trainWekaModel(getCLIJ2(), (ClearCLBuffer)( args[0]), (ClearCLBuffer)(args[1]), (String)args[2]);
        return true;
    }

    public static CLIJxWeka2 trainWekaModel(CLIJ2 clij2, ClearCLBuffer srcFeatureStack3D, ClearCLBuffer srcGroundTruth2D, String saveModelFilename) {
        CLIJxWeka2 weka = new CLIJxWeka2(clij2, srcFeatureStack3D, srcGroundTruth2D);
        weka.saveClassifier(saveModelFilename);
        return weka;
    }

    @Override
    public String getParameterHelpText() {
        return "Image featureStack3D, Image groundTruth2D, String saveModelFilename";
    }

    @Override
    public String getDescription() {
        return "Trains a Weka model using functionality of Fijis Trainable Weka Segmentation plugin. \n\n" +
                "It takes a 3D feature stack (e.g. first plane original image, second plane blurred, third plane edge image)" +
                "and trains a Weka model. This model will be saved to disc.\n" +
                "The given groundTruth image is supposed to be a label map where pixels with value 1 represent class 1, " +
                "pixels with value 2 represent class 2 and so on. Pixels with value 0 will be ignored for training.";
    }

    @Override
    public String getAvailableForDimensions() {
        return "2D";
    }


    @Override
    public String getCategories() {
        return "Machine Learning, Segmentation";
    }
}
