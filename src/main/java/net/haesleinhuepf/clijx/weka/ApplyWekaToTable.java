package net.haesleinhuepf.clijx.weka;

import ij.measure.ResultsTable;
import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij.coremem.enums.NativeTypeEnum;
import net.haesleinhuepf.clij.macro.CLIJMacroPlugin;
import net.haesleinhuepf.clij.macro.CLIJOpenCLProcessor;
import net.haesleinhuepf.clij.macro.documentation.OffersDocumentation;
import net.haesleinhuepf.clij2.AbstractCLIJ2Plugin;
import net.haesleinhuepf.clij2.CLIJ2;
import org.scijava.plugin.Plugin;

import java.nio.FloatBuffer;


/**
 * Author: @haesleinhuepf
 *         March 2020
 */

@Plugin(type = CLIJMacroPlugin.class, name = "CLIJx_applyWekaToTable")
public class ApplyWekaToTable extends AbstractCLIJ2Plugin implements CLIJMacroPlugin, CLIJOpenCLProcessor, OffersDocumentation {

    @Override
    public boolean executeCL() {
        ResultsTable table = ResultsTable.getResultsTable();
        String predictionColumn = (String) args[0];
        String loadModelFilename = (String) args[1];

        return applyWekaToTable(getCLIJ2(), table, predictionColumn, loadModelFilename);
    }

    public static boolean applyWekaToTable(CLIJ2 clij2, ResultsTable table, String predictionColumn, String loadModelFilename) {
        return applyWekaToTable(clij2, table, predictionColumn, null, loadModelFilename);
    }
    public static boolean applyWekaToTable(CLIJ2 clij2, ResultsTable table, String predictionColumn, CLIJxWeka2 clijxWeka) {
        return applyWekaToTable(clij2, table, predictionColumn, clijxWeka, null);
    }
    private static boolean applyWekaToTable(CLIJ2 clij2, ResultsTable table, String predictionColumn, CLIJxWeka2 clijxWeka, String loadModelFilename) {
        ResultsTable sendToGPUTable = TrainWekaFromTable.reformatTable(table, null);


        ClearCLBuffer tableOnGPU = clij2.create(sendToGPUTable.getHeadings().length, sendToGPUTable.size());

        clij2.resultsTableToImage2D(tableOnGPU, sendToGPUTable);

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //table.show("My Results");

        ClearCLBuffer transposed1 = clij2.create(tableOnGPU.getHeight(), tableOnGPU.getWidth());
        ClearCLBuffer featureStack = clij2.create(tableOnGPU.getHeight(), 1, tableOnGPU.getWidth());
        clij2.transposeXY(tableOnGPU, transposed1);
        clij2.transposeYZ(transposed1, featureStack);

        if (clijxWeka == null) {
            clijxWeka = new CLIJxWeka2(clij2, featureStack, loadModelFilename);
        } else {
            clijxWeka.setFeatureStack(featureStack);
        }
        ClearCLBuffer result = clijxWeka.getClassification();

        float[] resultArray = new float[(int) result.getWidth()];
        FloatBuffer buffer = FloatBuffer.wrap(resultArray);

        result.writeTo(buffer, true);

        for (int i = 0; i < resultArray.length; i++) {
            table.setValue(predictionColumn, i, resultArray[i] );
        }

        clij2.release(tableOnGPU);
        clij2.release(transposed1);
        clij2.release(featureStack);


        return true;
    }

    @Override
    public ClearCLBuffer createOutputBufferFromSource(ClearCLBuffer input) {
        return getCLIJ2().create(new long[]{input.getWidth(), input.getHeight()}, NativeTypeEnum.Float);
    }

    @Override
    public String getParameterHelpText() {
        return "String predictionColumnName, String loadModelFilename";
    }

    @Override
    public String getDescription() {
        return "Applies a Weka model using functionality of Fijis Trainable Weka Segmentation plugin.\n" +
                "It takes a Results Table, sorts its columns by name alphabetically and uses it as extracted features (rows correspond to feature vectors) " +
                "and applies a pre-trained a Weka model. Take care that the table has been generated in the same" +
                "way as for training the model!";
    }

    @Override
    public String getAvailableForDimensions() {
        return "2D";
    }

}
