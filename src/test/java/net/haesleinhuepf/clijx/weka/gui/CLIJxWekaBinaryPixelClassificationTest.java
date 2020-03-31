package net.haesleinhuepf.clijx.weka.gui;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij2.CLIJ2;

import static org.junit.Assert.*;

public class CLIJxWekaBinaryPixelClassificationTest {

    public static void main(String... args) {
        String filename = "C:/structure/data/Kota/NPC_T01.tif";

        new ImageJ();

        ImagePlus imp = IJ.openImage(filename);
        //imp = new Duplicator().run(imp, 2,2,1,1,1,1);
        //imp.show();

        CLIJ2 clijx = CLIJ2.getInstance();
        imp.setC(2);
        ClearCLBuffer input = clijx.pushCurrentSlice(imp);
        /*ClearCLBuffer slice1 = clij2.create(input);
        ClearCLBuffer slice2 = clij2.create(input);
        ClearCLBuffer slice3 = clij2.create(input);

        clij2.flip(input, slice1, true, false);
        clij2.flip(input, slice2, true, true);
        clij2.flip(input, slice3, false, true);

        ClearCLBuffer */

        clijx.show(input,"input");

        new CLIJxWekaBinaryPixelClassification().run(null);
    }


}