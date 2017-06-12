const Ganglion = require('../../OpenBCI_NodeJS_Ganglion/index').Ganglion;
const ganglion = new Ganglion();
var origin = 0;
var cpt = 0;
var start = new Date().getTime();

ganglion.once('ganglionFound', (peripheral) => {
  // Stop searching for BLE devices once a ganglion is found.
  ganglion.searchStop();
  ganglion.on('sample', (sample) => {
    /** Work with sample */
    console.log(sample.sampleNumber);
    // if (sample.sampleNumber == 180 || sample.sampleNumber == 179 || sample.sampleNumber == 177) {
      var end = new Date().getTime();
      // console.log((end-start)/1000);

      start = end;
    // }

    /*if (sample.sampleNumber == 0) {
      //create an array to store data for 1 second
      //console.log("origin", origin);
      origin = sample.channelData;
    }
    */

    for (let i = 0; i < ganglion.numberOfChannels(); i++) {
      console.log(sample.channelData[i].toFixed(8)); //+ " Volts.");
    }
  });
  ganglion.once('ready', () => {
    ganglion.streamStart();
  });
  ganglion.connect(peripheral);
});
// Start scanning for BLE devices
ganglion.searchStart();
