package net.haesleinhuepf.clijx.weka;

import ij.measure.ResultsTable;
import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij.macro.CLIJMacroPlugin;
import net.haesleinhuepf.clij.macro.CLIJOpenCLProcessor;
import net.haesleinhuepf.clij.macro.documentation.OffersDocumentation;
import net.haesleinhuepf.clij2.AbstractCLIJ2Plugin;
import net.haesleinhuepf.clij2.CLIJ2;
import net.haesleinhuepf.clij2.utilities.IsCategorized;
import org.scijava.plugin.Plugin;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;


/**
 * Author: @haesleinhuepf
 *         March 2020
 */

@Plugin(type = CLIJMacroPlugin.class, name = "CLIJx_trainWekaFromTable")
public class TrainWekaFromTable extends AbstractCLIJ2Plugin implements CLIJMacroPlugin, CLIJOpenCLProcessor, OffersDocumentation, IsCategorized {

    @Override
    public boolean executeCL() {
        ResultsTable table = ResultsTable.getResultsTable();
        String groundTruthColumn = (String) args[0];
        String saveModelFilename = (String) args[1];
        int numberOfTrees = asInteger(args[2]);
        int numberOfFeatures = asInteger(args[3]);
        int maxDepth = asInteger(args[4]);

        trainWekaFromTable(getCLIJ2(), table, groundTruthColumn, saveModelFilename, numberOfTrees, numberOfFeatures, maxDepth);
        return true;
    }

    public static CLIJxWeka2 trainWekaFromTable(CLIJ2 clij2, ResultsTable table, String groundTruthColumn, Integer numberOfTrees, Integer numberOfFeatures, Integer maxDepth) {
        return trainWekaFromTable(clij2, table, groundTruthColumn, null, numberOfTrees, numberOfFeatures, maxDepth);
    }

    public static CLIJxWeka2 trainWekaFromTable(CLIJ2 clij2, ResultsTable table, String groundTruthColumn, String saveModelFilename, Integer numberOfTrees, Integer numberOfFeatures, Integer maxDepth) {

        ResultsTable sendToGPUTable = reformatTable(table, groundTruthColumn);
        //sendToGPUTable.show("Send to GPU");


        ClearCLBuffer tableOnGPU = clij2.create(sendToGPUTable.getHeadings().length, sendToGPUTable.size());

        clij2.resultsTableToImage2D(tableOnGPU, sendToGPUTable);

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //table.show("My Results");

        ClearCLBuffer transposed1 = clij2.create(tableOnGPU.getHeight(), tableOnGPU.getWidth());
        ClearCLBuffer transposed2 = clij2.create(tableOnGPU.getHeight(), 1, tableOnGPU.getWidth());
        clij2.transposeXY(tableOnGPU, transposed1);
        clij2.transposeYZ(transposed1, transposed2);

        ClearCLBuffer ground_truth = clij2.create(transposed2.getWidth(), transposed2.getHeight(), 1);
        ClearCLBuffer featureStack = clij2.create(transposed2.getWidth(), transposed2.getHeight(), transposed2.getDepth() - 1);

        clij2.crop3D(transposed2, featureStack, 0, 0, 1);
        clij2.crop3D(transposed2, ground_truth, 0, 0, 0);

        //System.out.println("Ground truth");
        //clijx.print(ground_truth);


        CLIJxWeka2 weka = new CLIJxWeka2(clij2, featureStack, ground_truth);
        weka.setNumberOfTrees(numberOfTrees);
        weka.setNumberOfFeatures(numberOfFeatures);
        weka.setMaxDepth(maxDepth);
        System.out.println("Saved to " + saveModelFilename);
        weka.getClassifier();
        if (saveModelFilename != null) {
            weka.saveClassifier(saveModelFilename);
        }
        clij2.release(tableOnGPU);
        clij2.release(transposed1);
        clij2.release(transposed2);
        clij2.release(featureStack);
        clij2.release(ground_truth);

        return weka;
    }

    static ResultsTable reformatTable(ResultsTable table, String groundTruthColumn) {
        ResultsTable sendToGPUTable = new ResultsTable();

        ArrayList<String> columnNameList = new ArrayList<String>(Arrays.asList(table.getHeadings()));
        Collections.sort(columnNameList);

        // reformat the table so that ground truth is in the first colum
        // and all other colums are alphabetically sorted to the right
        for (int i = 0; i < table.size(); i++) {
            //System.out.println("i " + i);
            sendToGPUTable.incrementCounter();

            if (groundTruthColumn != null) {
                int ground_truth = (int) table.getValue(groundTruthColumn, i);
                sendToGPUTable.setValue(groundTruthColumn, i, ground_truth);
            }

            for (String column : columnNameList) {
                if (groundTruthColumn == null || column.compareTo(groundTruthColumn) != 0) {
                    sendToGPUTable.setValue(column, i, table.getValue(column, i));
                }
            }
        }

        return sendToGPUTable;
    }

    @Override
    public String getParameterHelpText() {
        return "String groundTruthColumnName, String saveModelFilename, Number trees, Number features, Number maxDepth";
    }

    @Override
    public String getDescription() {
        return "Trains a Weka model using functionality of Fijis Trainable Weka Segmentation plugin. \n\n" +
                "It takes the given Results Table, sorts its columns alphabetically as extracted features (rows correspond to feature vectors) and a " +
                "given column name containing the ground truth " +
                "to train a Weka model. This model will be saved to disc.\n" +
                "The given groundTruth column is supposed to be numeric with values 1 represent class 1, " +
                " value 2 represent class 2 and so on. Value 0 will be ignored for training.\n\n" +
                "Default values for options are:\n" +
                "* trees = 200\n" +
                "* features = 2\n" +
                "* maxDepth = 0";
    }

    @Override
    public String getAvailableForDimensions() {
        return "Table";
    }


    @Override
    public String getCategories() {
        return "Machine Learning, Segmentation";
    }
}
