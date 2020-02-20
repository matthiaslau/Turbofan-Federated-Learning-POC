// Script to request the stats from each engine and combine the results

let NUMBER_OF_NODES = 5

var failures = 0
var prevented_failures = 0
var prevented_failures_too_early = 0

for (var i = 0; i < NUMBER_OF_NODES; i++) {
    var port_suffix = ("0" + (i + 1)).slice(-2);
    var response = await axios.get('http://localhost:80' + port_suffix + '/stats')
    var data = await Promise.resolve(response)
    data = data['data']
    failures += data['FAILURES']
    prevented_failures += data['PREVENTED_FAILURES']
    prevented_failures_too_early += data['PREVENTED_FAILURES_TOO_EARLY']
}

console.log('FAILURES: ' + failures)
console.log('PREVENTED FAILURES: ' + prevented_failures)
console.log('PREVENTED FAILURES TOO EARLY: ' + prevented_failures_too_early)
