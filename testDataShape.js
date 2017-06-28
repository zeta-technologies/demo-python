const Ganglion = require('./OBGanglion/index').Ganglion;
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
    console.log(sample.channelData[0].toFixed(8)); //should be in microvolts

  });
  ganglion.once('ready', () => {
    ganglion.streamStart();
  });
  ganglion.connect(peripheral);
});
// Start scanning for BLE devices
ganglion.searchStart();
