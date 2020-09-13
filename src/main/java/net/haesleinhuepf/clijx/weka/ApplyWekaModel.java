package net.haesleinhuepf.clijx.weka;

import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij.coremem.enums.NativeTypeEnum;
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

@Plugin(type = CLIJMacroPlugin.class, name = "CLIJx_applyWekaModel")
public class ApplyWekaModel extends AbstractCLIJ2Plugin implements CLIJMacroPlugin, CLIJOpenCLProcessor, OffersDocumentation, ProcessableInTiles, IsCategorized {

    @Override
    public boolean executeCL() {
        applyWekaModel(getCLIJ2(), (ClearCLBuffer)( args[0]), (ClearCLBuffer)(args[1]), (String)args[2]);
        return true;
    }

    public static CLIJxWeka2 applyWekaModel(CLIJ2 clij2, ClearCLBuffer srcFeatureStack3D, ClearCLBuffer dstClassificationResult, String loadModelFilename) {
        CLIJxWeka2 weka = new CLIJxWeka2(clij2, srcFeatureStack3D, loadModelFilename);
        applyWekaModel(clij2, srcFeatureStack3D, dstClassificationResult, weka);
        return weka;
    }

    public static boolean applyWekaModel(CLIJ2 clij2, ClearCLBuffer srcFeatureStack3D, ClearCLBuffer dstClassificationResult, CLIJxWeka2 weka) {
        weka.setFeatureStack(srcFeatureStack3D);
        ClearCLBuffer classification = weka.getClassification();
        clij2.copy(classification, dstClassificationResult);
        clij2.release(classification);
        return true;
    }

    @Override
    public ClearCLBuffer createOutputBufferFromSource(ClearCLBuffer input) {
        return getCLIJ2().create(new long[]{input.getWidth(), input.getHeight()}, NativeTypeEnum.Float);
    }

    @Override
    public String getParameterHelpText() {
        return "Image featureStack3D, Image prediction2D_destination, String loadModelFilename";
    }

    @Override
    public String getDescription() {
        return "Applies a Weka model using functionality of Fijis Trainable Weka Segmentation plugin. \n\n" +
                "It takes a 3D feature stack (e.g. first plane original image, second plane blurred, third plane edge image)" +
                "and applies a pre-trained a Weka model. Take care that the feature stack has been generated in the same" +
                "way as for training the model!";
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
