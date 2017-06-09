//
//  ViewController.swift
//  MachineLearningClassifier
//
//  Created by Charles Scalesse on 6/7/17.
//  Copyright © 2017 Charles Scalesse. All rights reserved.
//

import UIKit
import AVFoundation
import CoreML
import Vision

class ViewController: UIViewController {
    private var session: AVCaptureSession?
    private var output: AVCaptureVideoDataOutput?
    private var previewLayer: AVCaptureVideoPreviewLayer?
    private var classificationRequest: VNCoreMLRequest?
    
    private let resultLabel: UILabel = {
        let label = UILabel(frame: CGRect(x: 0.0, y: 0.0, width: 0.0, height: 44.0))
        label.backgroundColor = .black
        label.textColor = .white
        label.font = .boldSystemFont(ofSize: 15.0)
        label.numberOfLines = 2
        label.textAlignment = .center
        return label
    }()
    
    // MARK: - Camera Methods
    
    private func initializeCamera() {
        session = AVCaptureSession()
        session?.sessionPreset = AVCaptureSession.Preset.high
        if let device = AVCaptureDevice.default(for: AVMediaType.video),
            let input = try? AVCaptureDeviceInput(device: device) {
            session?.addInput(input)
        }
        
        // use a background queue when processing the session output
        let queue = DispatchQueue.global(qos: .background)
        output = AVCaptureVideoDataOutput()
        output?.setSampleBufferDelegate(self, queue: queue)
        if let output = output {
            session?.addOutput(output);
        }
    }
    
    // MARK: - Vision Methods
    
    private func initializeVision() {
        guard let visionModel = try? VNCoreMLModel(for: Inceptionv3().model) else {
            fatalError("Couldn't load InceptionV3 model.")
        }
        
        classificationRequest = VNCoreMLRequest(model: visionModel, completionHandler: handleClassification)
        classificationRequest?.imageCropAndScaleOption = VNImageCropAndScaleOptionCenterCrop
    }
    
    func handleClassification(request: VNRequest, error: Error?) {
        guard let observations = request.results else {
            DispatchQueue.main.async {
                self.resultLabel.text = ""
            }
            return
        }
        
        let result = observations[0...4]                        // limit to 5 results
            .flatMap({ $0 as? VNClassificationObservation })
            .filter({ $0.confidence > 0.3 })                    // and at least 30% confidence
            .map({ $0.identifier })
            .joined(separator: ", ")
        
        DispatchQueue.main.async {
            self.resultLabel.text = result
        }
    }
    
    // MARK: - View Lifecycle
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        guard UIImagePickerController.isSourceTypeAvailable(.camera) else {
            let cameraLabel = UILabel(frame: self.view.bounds)
            cameraLabel.text = "This demo only works on a device with a camera."
            cameraLabel.backgroundColor = .black
            cameraLabel.textColor = .white
            cameraLabel.font = .boldSystemFont(ofSize: 15.0)
            cameraLabel.numberOfLines = 0
            cameraLabel.textAlignment = .center
            self.view.addSubview(cameraLabel)
            return
        }
        
        // initialize capture session & output
        initializeCamera()
        
        // initialize vision
        initializeVision()
        
        // add the preview layer
        if let session = session {
            previewLayer = AVCaptureVideoPreviewLayer(session: session)
            if let previewLayer = previewLayer {
                previewLayer.videoGravity = AVLayerVideoGravity.resizeAspectFill
                previewLayer.frame = self.view.bounds
                self.view.layer.addSublayer(previewLayer)
            }
        }
        
        // add the result label
        self.view.addSubview(resultLabel)
        
        // start the session
        session?.startRunning()
    }
    
    override func viewWillLayoutSubviews() {
        super.viewWillLayoutSubviews()
        previewLayer?.frame = self.view.bounds
        resultLabel.frame = CGRect(x: 0.0, y: self.view.frame.size.height - resultLabel.frame.size.height, width: self.view.frame.size.width, height: resultLabel.frame.size.height)
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate Methods

extension ViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer), let request = classificationRequest else {
            return
        }
        
        var requestOptions: [VNImageOption: Any] = [:]
        if let cameraIntrinsicData = CMGetAttachment(sampleBuffer, kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix, nil) {
            requestOptions = [.cameraIntrinsics: cameraIntrinsicData]
        }
        
        // @TODO: Get the orientation from the device. This only supports portrait orientation right now (6).
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: 6, options: requestOptions)
        do {
            try imageRequestHandler.perform([request])
        } catch {
            print(error)
        }
    }
    
}
