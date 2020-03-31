package net.haesleinhuepf.clijx.weka;

import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij.coremem.enums.NativeTypeEnum;
import net.haesleinhuepf.clij.macro.CLIJMacroPlugin;
import net.haesleinhuepf.clij.macro.CLIJOpenCLProcessor;
import net.haesleinhuepf.clij.macro.documentation.OffersDocumentation;
import net.haesleinhuepf.clij2.AbstractCLIJ2Plugin;
import net.haesleinhuepf.clij2.CLIJ2;
import org.jocl.CL;
import org.scijava.plugin.Plugin;

import java.io.File;
import java.util.HashMap;

/**
 * Author: @haesleinhuepf
 *         March 2020
 */

// @Plugin(type = CLIJMacroPlugin.class, name = "CLIJx_generateWekaProbabilityMaps")
public class GenerateWekaProbabilityMaps extends AbstractCLIJ2Plugin implements CLIJMacroPlugin, CLIJOpenCLProcessor, OffersDocumentation {

    @Override
    public boolean executeCL() {
        return generateWekaProbabilityMaps(getCLIJ2(), (ClearCLBuffer)( args[0]), (ClearCLBuffer)(args[1]), (String)args[2]);
    }

    public static boolean generateWekaProbabilityMaps(CLIJ2 clij2, ClearCLBuffer srcFeatureStack3D, ClearCLBuffer dstProbabilityMaps, String loadModelFilename) {
        /*if (new File(loadModelFilename + ".cl").exists()) {
            HashMap<String, Object> parameters = new HashMap<>();
            parameters.put("src_featureStack", srcFeatureStack3D);
            parameters.put("dst", dstProbabilityMaps);
            parameters.put("export_probabilities", 1);
            long[] dims = new long[]{dstProbabilityMaps.getWidth(), dstProbabilityMaps.getHeight()};
            clijx.execute(Object.class,loadModelFilename + ".cl", "classify_feature_stack", dims, dims, parameters);
        } else {
            new IllegalArgumentException("This model hasn't been saved as CLIJ OpenCL Model. Try retraining.");
        }*/
        CLIJxWeka2 clijxweka = new CLIJxWeka2(clij2, srcFeatureStack3D, loadModelFilename);
        ClearCLBuffer buffer = clijxweka.getDistribution();
        clij2.copy(buffer,dstProbabilityMaps);
        clij2.release(buffer);

        return true;
    }

    @Override
    public ClearCLBuffer createOutputBufferFromSource(ClearCLBuffer input) {
        String filename = (String) args[2];
        int numClasses = new CLIJxWeka2(null, null, filename).getNumberOfClasses();
        return getCLIJ2().create(new long[]{input.getWidth(), input.getHeight(), numClasses}, NativeTypeEnum.Float);
    }

    @Override
    public String getParameterHelpText() {
        return "Image featureStack3D, Image probabilityMap3D_destination, String loadModelFilename";
    }

    @Override
    public String getDescription() {
        return "Applies a Weka model which was saved as OpenCL file. Train your model with trainWekaModel to save it as OpenCL file.\n" +
                "It takes a 3D feature stack and applies a pre-trained a Weka model to produce probability maps for all classes. \n" +
                "Take care that the feature stack has been generated in the same way as for training the model!";
    }

    @Override
    public String getAvailableForDimensions() {
        return "2D";
    }
}
