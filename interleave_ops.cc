#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("Interleave")
    .Attr("T: {int32, float, double}")
    .Input("in1: T")
    .Input("in2: T")
    .Input("in3: T")
    .Input("in4: T")
    .Output("out: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle input;
      
      TF_RETURN_IF_ERROR( c->WithRank( c->input(0), 4, &input ) );
      
      ::tensorflow::shape_inference::DimensionHandle h_dim;
      c->Multiply( c->Dim( c->input(0), 1 ), 2, &h_dim );
      
      ::tensorflow::shape_inference::DimensionHandle w_dim;
      c->Multiply( c->Dim( c->input(0), 2 ), 2, &w_dim );
      
      input = c->MakeShape( { c->Dim( c->input(0), 0), h_dim, w_dim, c->Dim( c->input(0), 3 ) } );
      
      c->set_output(0, input);
      return Status::OK();
    });
