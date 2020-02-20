
var getUrl = window.location;
var baseUrl = getUrl.protocol + "//" + getUrl.host + "/" + getUrl.pathname.split('/')[1];

// AXIOS API Functions:
axios.baseUrl = baseUrl

async function get_engine_info() {
  try {
    const response = await axios.get('/info');
    return Promise.resolve(response)
  } catch (error) {
    console.error(error);
    return Promise.resolve(error)
  }
}

async function get_cycle_info() {
  try {
    const response = await axios.get('/cycle-info');
    return Promise.resolve(response)
  } catch (error) {
    console.error(error);
    return Promise.resolve(error)
  }
}

async function get_stats() {
  try {
    const response = await axios.get('/stats');
    return Promise.resolve(response)
  } catch (error) {
    console.error(error);
    return Promise.resolve(error)
  }
}

async function get_sensor_data() {
  try {
    const response = await axios.get('/sensor-data');
    return Promise.resolve(response)
  } catch (error) {
    console.error(error);
    return Promise.resolve(error)
  }
}

// Vue Variable Setters:

function set_state(new_state) {
  state.state = new_state
}

function set_engine_id(id) {
  engine_id.id = id
}

function set_grid_node_address(address) {
  grid_node.address = address
}

function set_grid_gateway_address(address) {
  grid_gateway.address = address
}

function set_failures(failure_count) {
  failures.count = failure_count
}

function set_prevented_failures(prevented_failures_count) {
  prevented_failures.count = prevented_failures_count
}

function set_maintenance_too_early(maintenance_too_early_count) {
  maintenance_too_early.count = maintenance_too_early_count
}

function set_cycle(current_cycle) {
  cycle.current = current_cycle
}

function set_predicted_rul(predicted_rul) {
  if (predicted_rul == null) predicted_rul = '-'
  prediction.rul = predicted_rul
}

// VUE OBJECT BINDINGS:

var engine_id = new Vue({
  el: '#engine',
  delimiters: ['[[', ']]'],
  data: {
    id: ""
  }
});

var state = new Vue({
  el: '#state',
  delimiters: ['[[', ']]'],
  data: {
    state: ""
  }
});

var grid_node = new Vue({
  el: '#grid-node',
  delimiters: ['[[', ']]'],
  data: {
    address: ""
  }
});

var grid_gateway = new Vue({
  el: '#grid-gateway',
  delimiters: ['[[', ']]'],
  data: {
    address: ""
  }
});

var failures = new Vue({
  el: '#failures',
  delimiters: ['[[', ']]'],
  data: {
    count: ""
  }
});

var prevented_failures = new Vue({
  el: '#prevented-failures',
  delimiters: ['[[', ']]'],
  data: {
    count: ""
  }
});

var maintenance_too_early = new Vue({
  el: '#maintenance-too-early',
  delimiters: ['[[', ']]'],
  data: {
    count: ""
  }
});

var cycle = new Vue({
  el: '#cycle',
  delimiters: ['[[', ']]'],
  data: {
    current: ""
  }
});

var prediction = new Vue({
  el: '#prediction',
  delimiters: ['[[', ']]'],
  data: {
    rul: ""
  }
});

// DIAGRAM LOGIC

let displayed_sensors = [
    "operational_setting_1",
    "operational_setting_2",
    "operational_setting_3",
    "sensor_measurement_1",
    "sensor_measurement_2",
    "sensor_measurement_3",
    "sensor_measurement_4",
    "sensor_measurement_5",
    "sensor_measurement_6",
    "sensor_measurement_7",
    "sensor_measurement_8",
    "sensor_measurement_9",
    "sensor_measurement_10",
    "sensor_measurement_11",
    "sensor_measurement_12",
    "sensor_measurement_13",
    "sensor_measurement_14",
    "sensor_measurement_15",
    "sensor_measurement_16",
    "sensor_measurement_17",
    "sensor_measurement_18",
    "sensor_measurement_19",
    "sensor_measurement_20",
    "sensor_measurement_21",
]
var charts = []

function init_diagrams() {
    displayed_sensors.forEach((sensor) => {
        var chart_node = document.createElement("div")
        //chart_node.setAttribute('class', 'uk-')
        var canvas_node = document.createElement("canvas")
        canvas_node.id = sensor
        canvas_node.setAttribute('width', '250')
        canvas_node.setAttribute('height', '100')
        chart_node.appendChild(canvas_node)
        document.getElementById("charts-container").appendChild(chart_node)

        var ctx = canvas_node.getContext('2d')
        charts[sensor] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: sensor,
                    data: [],
                    borderWidth: 2,
                    borderColor: '#ffffff',
                    pointRadius: 0,
                    fill: false
                }]
            },
            options: {
                scales: {
                    xAxes: [{
                        gridLines: {
                            color: 'rgba(254, 254, 254, 0.1)',
                            zeroLineWidth: 0
                        },
                        ticks: {
                            display: false
                        },
                     }],
                     yAxes: [{
                        angleLines: {
                            display: false
                        },
                        gridLines: {
                            color: 'rgba(254, 254, 254, 0.1)',
                            zeroLineWidth: 0
                        },
                        ticks: {
                            display: false
                        }
                     }]
                },
                legend: {
                    labels: {
                        boxWidth: 0,
                        fontColor: '#ffffff'
                    }
                }
            }
        })
    })
}

function update_diagrams(data) {
    displayed_sensors.forEach((sensor) => {
        let sensor_values = data[sensor]
        var chart = charts[sensor]
        if (chart != null) {
            let cycles = Array((sensor_values || []).length).fill(0).map((e,i)=>i+1)
            chart.data.labels = cycles
            chart.data.datasets[0].data = sensor_values
            chart.update()
        }
    })
}

// MAIN LOGIC

async function update_engine_info() {
  var info = await get_engine_info()
  set_engine_id(info["data"]["id"])
  set_grid_node_address(info["data"]["grid_node_address"])
  set_grid_gateway_address(info["data"]["grid_gateway_address"])
}

async function update_cycle_info() {
  var cycle_info = await get_cycle_info()
  set_state(cycle_info["data"]["state"])
  set_cycle(cycle_info["data"]["cycle"])
  set_predicted_rul(cycle_info["data"]["predicted_rul"])
}

async function update_stats() {
  var stats = await get_stats()
  set_failures(stats["data"]["FAILURES"])
  set_prevented_failures(stats["data"]["PREVENTED_FAILURES"])
  set_maintenance_too_early(stats["data"]["PREVENTED_FAILURES_TOO_EARLY"])
}

async function update_sensor_data() {
  let response = await get_sensor_data()
  let sensor_data = response["data"]["sensor_data"]

  var data = []

  sensor_data.forEach((cycle) => {
    for (var key in cycle) {
        if (data[key] == undefined) {
            data[key] = []
        }
        data[key].push(cycle[key])
    }
  });

  update_diagrams(data)
}

async function sync_with_server() {
  await update_cycle_info()
  await update_stats()
  await update_sensor_data()
  setTimeout(sync_with_server, 1000)
}

update_engine_info()
init_diagrams()
sync_with_server()
