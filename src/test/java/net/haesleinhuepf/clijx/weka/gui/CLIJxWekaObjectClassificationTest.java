package net.haesleinhuepf.clijx.weka.gui;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij2.CLIJ2;

import static org.junit.Assert.*;

public class CLIJxWekaObjectClassificationTest {


    public static void main(String... args) {
        String filename = "C:/structure/data/Kota/NPC_T01.tif";

        new ImageJ();

        ImagePlus imp = IJ.openImage(filename);
        //imp = new Duplicator().run(imp, 2,2,1,1,1,1);
        //imp.show();

        CLIJ2 clijx = CLIJ2.getInstance();
        imp.setC(2);
        ClearCLBuffer input = clijx.pushCurrentSlice(imp);
        clijx.show(input,"input");
        ClearCLBuffer thresholded = clijx.create(input);
        ClearCLBuffer temp = clijx.create(input);
        clijx.gaussianBlur2D(input, temp, 3,3);
        clijx.thresholdOtsu(temp, thresholded);
        clijx.show(thresholded, "thresholded");
        clijx.clear();

        new CLIJxWekaObjectClassification().run(null);
    }

}