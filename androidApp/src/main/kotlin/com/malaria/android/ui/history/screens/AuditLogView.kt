package com.malaria.android.ui.history.screens

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.malaria.android.data.entities.AuditAction
import com.malaria.android.ui.history.HistoryDestination
import com.malaria.android.ui.history.HistoryNavigator
import com.malaria.android.ui.history.components.AuditEntryRowView
import com.malaria.android.ui.locals.LocalDatabase
import kotlinx.datetime.Clock
import kotlinx.datetime.DateTimeUnit
import kotlinx.datetime.Instant
import kotlinx.datetime.LocalDate
import kotlinx.datetime.TimeZone
import kotlinx.datetime.atStartOfDayIn
import kotlinx.datetime.minus
import kotlinx.datetime.toLocalDateTime

/**
 * Audit-log viewer with action + date-range filters. Mirrors
 * `iosApp/Views/History/AuditLogView.swift` (spec §11). Filters are applied
 * client-side over the flow of recent 200 entries; "Load older" lands in
 * a later phase.
 *
 * Picker is a standalone Material 3 [DropdownMenu] rather than
 * [androidx.compose.material3.ExposedDropdownMenuBox] — same Phase 7
 * decision driven by Compose BOM API drift on the experimental box.
 */
@Composable
fun AuditLogView(@Suppress("UNUSED_PARAMETER") navigator: HistoryNavigator) {
    val dao = LocalDatabase.current.auditDao()
    val flow = remember { dao.recentFlow(limit = 200) }
    val allEntries by flow.collectAsState(initial = emptyList())

    var selectedAction by remember { mutableStateOf<AuditAction?>(null) }
    var useDateFilter by remember { mutableStateOf(false) }
    val today = Clock.System.now()
    var fromInstant by remember {
        mutableStateOf(today.minus(30, DateTimeUnit.DAY, TimeZone.currentSystemDefault()))
    }
    var toInstant by remember { mutableStateOf(today) }

    val filtered = allEntries.asSequence()
        .filter { entry ->
            val matchAction = selectedAction?.let { it.canonical == entry.action } ?: true
            val matchDate = if (useDateFilter) {
                entry.timestamp >= fromInstant && entry.timestamp <= toInstant
            } else {
                true
            }
            matchAction && matchDate
        }
        .take(200)
        .toList()

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(start = 16.dp, end = 16.dp),
    ) {
        item {
            FilterSection(
                selectedAction = selectedAction,
                onActionChange = { selectedAction = it },
                useDateFilter = useDateFilter,
                onUseDateFilterChange = { useDateFilter = it },
                fromInstant = fromInstant,
                onFromInstantChange = { fromInstant = it },
                toInstant = toInstant,
                onToInstantChange = { toInstant = it },
            )
            HorizontalDivider(modifier = Modifier.padding(top = 12.dp, bottom = 12.dp))
        }

        if (filtered.isEmpty()) {
            item {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 24.dp, bottom = 24.dp),
                    contentAlignment = Alignment.Center,
                ) {
                    Text(
                        text = "No matching entries",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        } else {
            item {
                Text(
                    text = "Entries (${filtered.size})",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(bottom = 4.dp),
                )
            }
            items(filtered, key = { it.id }) { entry ->
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable {
                            navigator.push(HistoryDestination.AuditEntryDetail(entry.id))
                        }
                        .padding(top = 4.dp, bottom = 4.dp),
                ) {
                    AuditEntryRowView(entry = entry)
                    HorizontalDivider(modifier = Modifier.padding(top = 4.dp))
                }
            }
        }
    }
}

@Composable
private fun FilterSection(
    selectedAction: AuditAction?,
    onActionChange: (AuditAction?) -> Unit,
    useDateFilter: Boolean,
    onUseDateFilterChange: (Boolean) -> Unit,
    fromInstant: Instant,
    onFromInstantChange: (Instant) -> Unit,
    toInstant: Instant,
    onToInstantChange: (Instant) -> Unit,
) {
    Column(
        modifier = Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text(
            text = "Filter",
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 12.dp, bottom = 4.dp),
        )
        ActionDropdown(selected = selectedAction, onChange = onActionChange)
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = "Date range",
                style = MaterialTheme.typography.bodyMedium,
                modifier = Modifier.weight(1f),
            )
            Switch(
                checked = useDateFilter,
                onCheckedChange = onUseDateFilterChange,
            )
        }
        if (useDateFilter) {
            DateField(
                label = "From",
                instant = fromInstant,
                onInstantChange = onFromInstantChange,
            )
            DateField(
                label = "To",
                instant = toInstant,
                onInstantChange = onToInstantChange,
            )
        }
    }
}

@Composable
private fun ActionDropdown(
    selected: AuditAction?,
    onChange: (AuditAction?) -> Unit,
) {
    var expanded by remember { mutableStateOf(false) }

    Box(modifier = Modifier.fillMaxWidth()) {
        OutlinedTextField(
            value = selected?.canonical ?: "All",
            onValueChange = {},
            readOnly = true,
            label = { Text("Action") },
            trailingIcon = {
                IconButton(onClick = { expanded = !expanded }) {
                    Icon(Icons.Filled.ArrowDropDown, contentDescription = "Open action list")
                }
            },
            modifier = Modifier.fillMaxWidth(),
        )
        DropdownMenu(
            expanded = expanded,
            onDismissRequest = { expanded = false },
        ) {
            DropdownMenuItem(
                text = { Text("All") },
                onClick = {
                    onChange(null)
                    expanded = false
                },
            )
            AuditAction.entries.forEach { action ->
                DropdownMenuItem(
                    text = { Text(action.canonical) },
                    onClick = {
                        onChange(action)
                        expanded = false
                    },
                )
            }
        }
    }
}

/**
 * ISO date entry — a plain [OutlinedTextField] taking `YYYY-MM-DD`. A
 * full Material 3 DatePickerDialog is BOM-experimental; the typed field
 * keeps the surface stable across BOM revs and is enough for the v1
 * audit filter use case.
 */
@Composable
private fun DateField(
    label: String,
    instant: Instant,
    onInstantChange: (Instant) -> Unit,
) {
    val current = formatDate(instant)
    var text by remember(current) { mutableStateOf(current) }
    OutlinedTextField(
        value = text,
        onValueChange = { new ->
            text = new
            parseIsoDate(new)?.let(onInstantChange)
        },
        label = { Text("$label (YYYY-MM-DD)") },
        modifier = Modifier.fillMaxWidth(),
        singleLine = true,
    )
}

private fun formatDate(instant: Instant): String {
    val zone = TimeZone.currentSystemDefault()
    val local = instant.toLocalDateTime(zone)
    return "${local.year}-${local.monthNumber.toString().padStart(2, '0')}-" +
        local.dayOfMonth.toString().padStart(2, '0')
}

private fun parseIsoDate(text: String): Instant? {
    val parts = text.split('-')
    if (parts.size != 3) return null
    val year = parts[0].toIntOrNull() ?: return null
    val month = parts[1].toIntOrNull() ?: return null
    val day = parts[2].toIntOrNull() ?: return null
    return try {
        val date = LocalDate(year, month, day)
        date.atStartOfDayIn(TimeZone.currentSystemDefault())
    } catch (_: IllegalArgumentException) {
        null
    }
}
