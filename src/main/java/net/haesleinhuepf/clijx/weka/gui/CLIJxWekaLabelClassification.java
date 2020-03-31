package net.haesleinhuepf.clijx.weka.gui;

import fiji.util.gui.GenericDialogPlus;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.gui.Roi;
import ij.plugin.frame.RoiManager;
import net.haesleinhuepf.clij.CLIJ;
import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;

import java.awt.*;
import java.util.ArrayList;

public class CLIJxWekaLabelClassification extends CLIJxWekaObjectClassification {


    protected void generateROIs(ImagePlus imp) {

        ClearCLBuffer labelled = clij2.push(imp);

        RoiManager roiManager = new RoiManager(false);
        clij2.pullLabelsToROIManager(labelled, roiManager);

        for (int i = 0; i < roiManager.getCount(); i++) {
            Roi roi = roiManager.getRoi(i);
            roi.setStrokeColor(Color.white);
            //roi.setStrokeWidth(1);
            roi.setName("");
            overlay.add(roi);
        }
        inputImp.setOverlay(overlay);
    }

    protected boolean showInitialDialog() {
        GenericDialogPlus gd = new GenericDialogPlus("CLIJx Weka Label Classification");
        ArrayList<String> deviceNameList = CLIJ.getAvailableDeviceNames();
        String[] deviceNameArray = new String[deviceNameList.size()];
        deviceNameList.toArray(deviceNameArray);
        gd.addChoice("OpenCL Device", deviceNameArray, CLIJxWekaPropertyHolder.clDeviceName);
        gd.addImageChoice("Input image", IJ.getImage().getTitle());
        gd.addImageChoice("Label map", IJ.getImage().getTitle());
        gd.addNumericField("Number of object classes (minimum: 2)", CLIJxWekaPropertyHolder.numberOfObjectClasses, 0);
        gd.showDialog();

        if (gd.wasCanceled()) {
            return false;
        }


        CLIJxWekaPropertyHolder.clDeviceName = gd.getNextChoice();

        inputImp = gd.getNextImage();
        binaryImp = gd.getNextImage();

        CLIJxWekaPropertyHolder.numberOfObjectClasses = (int) gd.getNextNumber();
        return true;
    }

}
