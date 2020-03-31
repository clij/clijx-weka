package net.haesleinhuepf.clijx.weka.gui;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij2.CLIJ2;

import static org.junit.Assert.*;

public class CLIJxWekaLabelClassificationTest {


    public static void main(String... args) {
        String imageFilename = "C:/structure/data/2019-12-17-16-54-37-81-Lund_Tribolium_nGFP_TMR/processed/01_max_image/001200.raw.tif";
        String labelsFilename = "C:/structure/data/2019-12-17-16-54-37-81-Lund_Tribolium_nGFP_TMR/processed/07_max_cells/001200.raw.tif";

        new ImageJ();

        ImagePlus imp = IJ.openImage(imageFilename);
        ImagePlus lab = IJ.openImage(labelsFilename);
        IJ.run(imp,"Rotate 90 Degrees Right", "");
        IJ.run(lab,"Rotate 90 Degrees Right", "");

        CLIJ2 clijx = CLIJ2.getInstance();
        ClearCLBuffer input = clijx.push(imp);
        ClearCLBuffer labels = clijx.push(lab);

        clijx.show(input,"input");
        clijx.show(labels, "labels");
        clijx.clear();

        new CLIJxWekaLabelClassification().run(null);
    }
}