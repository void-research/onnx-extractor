use onnx_extractor::OnnxModel;

const MODEL_PATH: &str = "tests/mnist-12.onnx";

#[test]
fn load_mnist_model() {
    // use CARGO_MANIFEST_DIR so the test works from any working directory
    let path = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), MODEL_PATH);
    let model = OnnxModel::load_from_file(&path).expect("Failed to load mnist model");

    // basic sanity checks
    assert!(!model.inputs.is_empty(), "model should have inputs");
    assert!(!model.outputs.is_empty(), "model should have outputs");
    assert!(!model.operations.is_empty(), "model should have operations");
    assert!(!model.tensors.is_empty(), "model should have tensors");
}

#[test]
fn test_tensor_queries() {
    let path = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), MODEL_PATH);
    let model = OnnxModel::load_from_file(&path).expect("Failed to load mnist model");

    // tensor names should be non-empty and get_tensor should return for the first one
    let names = model.tensor_names();
    assert!(!names.is_empty(), "tensor_names should not be empty");
    let first_name = names[0];
    assert!(
        model.get_tensor(first_name).is_some(),
        "get_tensor should find the tensor"
    );
}

#[test]
fn test_operation_queries() {
    let path = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), MODEL_PATH);
    let model = OnnxModel::load_from_file(&path).expect("Failed to load mnist model");

    // operation types list should be non-empty and getting operations by type should work
    let op_types = model.operation_types();
    assert!(!op_types.is_empty(), "operation_types should not be empty");
    let first_type = &op_types[0];
    let ops_of_type = model.get_operations_by_type(first_type);
    assert!(
        !ops_of_type.is_empty(),
        "get_operations_by_type should return at least one op"
    );

    // get_operation for a real op name
    let first_op = &model.operations[0];
    let found = model.get_operation(&first_op.name);
    assert!(
        found.is_some(),
        "get_operation should return the operation by name"
    );
}

#[test]
fn test_input_output_and_weights() {
    let path = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), MODEL_PATH);
    let model = OnnxModel::load_from_file(&path).expect("Failed to load mnist model");

    let input_tensors = model.get_input_tensors();
    let output_tensors = model.get_output_tensors();
    assert!(
        !input_tensors.is_empty(),
        "get_input_tensors should return inputs"
    );
    assert!(
        !output_tensors.is_empty(),
        "get_output_tensors should return outputs"
    );

    // weight tensors may be empty for some models, but calling should not panic
    let _weights = model.get_weight_tensors();
}

#[test]
fn test_topological_order() {
    let path = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), MODEL_PATH);
    let model = OnnxModel::load_from_file(&path).expect("Failed to load mnist model");

    let ordered = model
        .topological_order()
        .expect("topological_order should succeed");
    // ordering should include every operation exactly once
    assert_eq!(
        ordered.len(),
        model.operations.len(),
        "topological order should include all operations"
    );

    // all names in ordered should be found in the original ops
    let orig_names: std::collections::HashSet<&str> =
        model.operations.iter().map(|o| o.name.as_str()).collect();
    for op in ordered {
        assert!(
            orig_names.contains(op.name.as_str()),
            "ordered op should exist in original operations"
        );
    }
}

#[test]
fn test_get_raw_data() {
    let path = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), MODEL_PATH);
    let model = OnnxModel::load_from_file(&path).expect("Failed to load mnist model");

    let weights = model.get_weight_tensors();
    // ensure at least one weight tensor exists in the model for this test
    assert!(
        !weights.is_empty(),
        "model should contain at least one weight tensor"
    );

    let first = weights[0];
    let data_ref = first.data().expect("data() should return tensor data");
    assert!(!data_ref.is_empty(), "tensor data should be non-empty");
}

#[test]
fn test_no_data_tensors_report_no_data() {
    let path = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), MODEL_PATH);
    let model = OnnxModel::load_from_file(&path).expect("Failed to load mnist model");

    let tensor = model
        .get_tensor("ReLU114_Output_0")
        .expect("ReLU114_Output_0 tensor should exist");

    assert!(
        tensor.data().is_err(),
        "ReLU114_Output_0 should not have embedded data and data() must error"
    );
}
