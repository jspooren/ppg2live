// Arduino project to read out the PulseSensor™ and send the raw values to the serial port.
// See http://www.pulsesensor.com for info on the pulse sensor.

//  Variables
int PulseSensorPurplePin = 0;        // Pulse Sensor PURPLE WIRE connected to ANALOG PIN 0
int LED13 = 13;                      // The on-board Arduino LED
int Signal;                          // The incoming raw data. Signal in range [0,1024]
int Threshold = 550;                 // Threshold to detect a heat beat 


// The SetUp Function:
void setup() {
   pinMode(LED13,OUTPUT);            // Blink LED to show heartbeat
   Serial.begin(57600);              // Initialize serial port to 57600 bps
}

// The Main Loop Function
void loop() {

   Signal = analogRead(PulseSensorPurplePin);  // Read the PulseSensor's value 
   Serial.println(Signal) ;                    // Send to serial port
   
   if(Signal > Threshold)            // Blink LED with detected heartbeat
   {
      digitalWrite(LED13,HIGH);          
   } 
   else 
   {
      digitalWrite(LED13,LOW);
   }

   delay(10);                        // Wait a bit
