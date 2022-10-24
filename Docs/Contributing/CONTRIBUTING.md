# Contributing

Thank you for you interest in contributing to MAKit! From commenting to 
reviewing and sending MR requests, all contributions are welcome. 

## Developer environment

Install XCode with command line tools.

## Coding style

Go to XCode preferences -> Text editing tab -> Page guide at column: 80.

## Testing

### CI 

The [unit tests](../Architecture/MAKitTests.md) 
are run after each push on the repository. 

The [integration tests](../Architecture/MATorchTests.md) 
are not run systematically, 
neither are the [examples](../Architecture/MAExamples.md).

Once the MR is "ready to review", please trigger the workflows on GitHub 
to ensure these additional tests have completed. 

### Local 

Testing the [unit tests](../Architecture/MAKitTests.md) 
on XCode is straight forward.

Testing the [integration tests](../Architecture/MATorchTests.md) 
and the [examples](../Architecture/MAExamples.md) require an additional setup. 
Follow the previous links to know more about.

## Release
