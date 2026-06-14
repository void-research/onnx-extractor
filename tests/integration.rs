use onnx_extractor::Model;

const MODEL_PATH: &str = "tests/mnist-12.onnx";

#[test]
fn load_mnist_model() {
    // use CARGO_MANIFEST_DIR so the test works from any working directory
    let path = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), MODEL_PATH);
    let model = Model::load_from_file(&path).expect("Failed to load mnist model");

    // basic sanity checks
    assert!(
        !model.graph().inputs().is_empty(),
        "model should have inputs"
    );
    assert!(
        !model.graph().outputs().is_empty(),
        "model should have outputs"
    );
    assert!(
        !model.graph().operations().is_empty(),
        "model should have operations"
    );
    assert!(
        model.graph().tensors().keys().next().is_some(),
        "model should have tensors"
    );
}

#[test]
fn test_tensor_queries() {
    let path = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), MODEL_PATH);
    let model = Model::load_from_file(&path).expect("Failed to load mnist model");

    // tensor names should be non-empty and get_tensor should return for the first one
    let mut names = model.graph().tensors().keys();
    let first_name = names.next().expect("tensor_names should not be empty");
    assert!(
        model.graph().tensors().get(first_name).is_some(),
        "get_tensor should find the tensor"
    );
}

#[test]
fn test_operation_queries() {
    let path = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), MODEL_PATH);
    let model = Model::load_from_file(&path).expect("Failed to load mnist model");

    // operation types list should be non-empty and getting operations by type should work
    let op_types = model.graph().operation_types();
    assert!(!op_types.is_empty(), "operation_types should not be empty");
    let first_type = &op_types[0];
    let mut ops_of_type = model.graph().get_operations_by_type(first_type);
    assert!(
        ops_of_type.next().is_some(),
        "get_operations_by_type should return at least one op"
    );

    // get_operation for a real op name
    let first_op = &model.graph().operations()[0];
    let found = model.graph().get_operation(first_op.name());
    assert!(
        found.is_some(),
        "get_operation should return the operation by name"
    );
}

#[test]
fn test_input_output_and_weights() {
    let path = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), MODEL_PATH);
    let model = Model::load_from_file(&path).expect("Failed to load mnist model");

    let mut input_tensors = model.graph().get_input_tensors();
    let mut output_tensors = model.graph().get_output_tensors();
    assert!(
        input_tensors.next().is_some(),
        "get_input_tensors should return inputs"
    );
    assert!(
        output_tensors.next().is_some(),
        "get_output_tensors should return outputs"
    );

    // weight tensors may be empty for some models, but calling should not panic
    let _weights = model.graph().get_weight_tensors();
}

#[test]
fn test_topological_order() {
    let path = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), MODEL_PATH);
    let model = Model::load_from_file(&path).expect("Failed to load mnist model");

    let ordered = model
        .graph()
        .topological_order()
        .expect("topological_order should succeed");
    // ordering should include every operation exactly once
    assert_eq!(
        ordered.len(),
        model.graph().operations().len(),
        "topological order should include all operations"
    );

    // all names in ordered should be found in the original ops
    let orig_names: std::collections::HashSet<&str> = model
        .graph()
        .operations()
        .iter()
        .map(|o| o.name())
        .collect();
    for op in ordered {
        assert!(
            orig_names.contains(op.name()),
            "ordered op should exist in original operations"
        );
    }
}

#[test]
fn test_get_raw_data() {
    let path = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), MODEL_PATH);
    let model = Model::load_from_file(&path).expect("Failed to load mnist model");

    let weights = model.graph().get_weight_tensors();
    let mut weights_iter = weights.into_iter();
    let first = weights_iter
        .next()
        .expect("model should contain at least one weight tensor");

    let data_ref = first.data().expect("data() should return tensor data");
    assert!(!data_ref.is_empty(), "tensor data should be non-empty");
}

#[test]
fn test_no_data_tensors_report_no_data() {
    let path = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), MODEL_PATH);
    let model = Model::load_from_file(&path).expect("Failed to load mnist model");

    let tensor = model
        .graph()
        .tensors()
        .get("ReLU114_Output_0")
        .expect("ReLU114_Output_0 tensor should exist");

    assert!(
        tensor.data().is_err(),
        "ReLU114_Output_0 should not have embedded data and data() must error"
    );
}
