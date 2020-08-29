/*
 * CloudViewerCellPicker.h
 *
 *  Created on: Aug 21, 2018
 *      Author: mathieu
 */

#ifndef CLOUDVIEWERCELLPICKER_H_
#define CLOUDVIEWERCELLPICKER_H_

#include <vtkCellPicker.h>

class CloudViewerCellPicker : public vtkCellPicker {
public:
public:
    static CloudViewerCellPicker *New ();
    vtkTypeMacro(CloudViewerCellPicker, vtkCellPicker)
	CloudViewerCellPicker();
    virtual ~CloudViewerCellPicker() override;

protected:
	// overrided to ignore back faces
    virtual double IntersectActorWithLine(const double p1[3],
    		const double p2[3],
			double t1, double t2,
			double tol,
			vtkProp3D *prop,
            vtkMapper *mapper) override;

private:
	vtkGenericCell * cell_; //used to accelerate picking
	vtkIdList * pointIds_; // used to accelerate picking
};

#endif /* CLOUDVIEWERCELLPICKER_H_ */
